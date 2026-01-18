import binascii
from collections import defaultdict, deque
from contextlib import suppress
from io import BytesIO, UnsupportedOperation
import os
import struct
import sys
from itertools import chain
from typing import (
import warnings
import zlib
from hashlib import sha1
from os import SEEK_CUR, SEEK_END
from struct import unpack_from
from .errors import ApplyDeltaError, ChecksumMismatch
from .file import GitFile
from .lru_cache import LRUSizeCache
from .objects import ObjectID, ShaFile, hex_to_sha, object_header, sha_to_hex
class PackData:
    """The data contained in a packfile.

    Pack files can be accessed both sequentially for exploding a pack, and
    directly with the help of an index to retrieve a specific object.

    The objects within are either complete or a delta against another.

    The header is variable length. If the MSB of each byte is set then it
    indicates that the subsequent byte is still part of the header.
    For the first byte the next MS bits are the type, which tells you the type
    of object, and whether it is a delta. The LS byte is the lowest bits of the
    size. For each subsequent byte the LS 7 bits are the next MS bits of the
    size, i.e. the last byte of the header contains the MS bits of the size.

    For the complete objects the data is stored as zlib deflated data.
    The size in the header is the uncompressed object size, so to uncompress
    you need to just keep feeding data to zlib until you get an object back,
    or it errors on bad data. This is done here by just giving the complete
    buffer from the start of the deflated object on. This is bad, but until I
    get mmap sorted out it will have to do.

    Currently there are no integrity checks done. Also no attempt is made to
    try and detect the delta case, or a request for an object at the wrong
    position.  It will all just throw a zlib or KeyError.
    """

    def __init__(self, filename, file=None, size=None) -> None:
        """Create a PackData object representing the pack in the given filename.

        The file must exist and stay readable until the object is disposed of.
        It must also stay the same size. It will be mapped whenever needed.

        Currently there is a restriction on the size of the pack as the python
        mmap implementation is flawed.
        """
        self._filename = filename
        self._size = size
        self._header_size = 12
        if file is None:
            self._file = GitFile(self._filename, 'rb')
        else:
            self._file = file
        version, self._num_objects = read_pack_header(self._file.read)
        self._offset_cache = LRUSizeCache[int, Tuple[int, OldUnpackedObject]](1024 * 1024 * 20, compute_size=_compute_object_size)

    @property
    def filename(self):
        return os.path.basename(self._filename)

    @property
    def path(self):
        return self._filename

    @classmethod
    def from_file(cls, file, size=None):
        return cls(str(file), file=file, size=size)

    @classmethod
    def from_path(cls, path):
        return cls(filename=path)

    def close(self):
        self._file.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __eq__(self, other):
        if isinstance(other, PackData):
            return self.get_stored_checksum() == other.get_stored_checksum()
        return False

    def _get_size(self):
        if self._size is not None:
            return self._size
        self._size = os.path.getsize(self._filename)
        if self._size < self._header_size:
            errmsg = '%s is too small for a packfile (%d < %d)' % (self._filename, self._size, self._header_size)
            raise AssertionError(errmsg)
        return self._size

    def __len__(self) -> int:
        """Returns the number of objects in this pack."""
        return self._num_objects

    def calculate_checksum(self):
        """Calculate the checksum for this pack.

        Returns: 20-byte binary SHA1 digest
        """
        return compute_file_sha(self._file, end_ofs=-20).digest()

    def iter_unpacked(self, *, include_comp: bool=False):
        self._file.seek(self._header_size)
        if self._num_objects is None:
            return
        for _ in range(self._num_objects):
            offset = self._file.tell()
            unpacked, unused = unpack_object(self._file.read, compute_crc32=False, include_comp=include_comp)
            unpacked.offset = offset
            yield unpacked
            self._file.seek(-len(unused), SEEK_CUR)

    def iterentries(self, progress=None, resolve_ext_ref: Optional[ResolveExtRefFn]=None):
        """Yield entries summarizing the contents of this pack.

        Args:
          progress: Progress function, called with current and total
            object count.
        Returns: iterator of tuples with (sha, offset, crc32)
        """
        num_objects = self._num_objects
        indexer = PackIndexer.for_pack_data(self, resolve_ext_ref=resolve_ext_ref)
        for i, result in enumerate(indexer):
            if progress is not None:
                progress(i, num_objects)
            yield result

    def sorted_entries(self, progress: Optional[ProgressFn]=None, resolve_ext_ref: Optional[ResolveExtRefFn]=None):
        """Return entries in this pack, sorted by SHA.

        Args:
          progress: Progress function, called with current and total
            object count
        Returns: Iterator of tuples with (sha, offset, crc32)
        """
        return sorted(self.iterentries(progress=progress, resolve_ext_ref=resolve_ext_ref))

    def create_index_v1(self, filename, progress=None, resolve_ext_ref=None):
        """Create a version 1 file for this data file.

        Args:
          filename: Index filename.
          progress: Progress report function
        Returns: Checksum of index file
        """
        entries = self.sorted_entries(progress=progress, resolve_ext_ref=resolve_ext_ref)
        with GitFile(filename, 'wb') as f:
            return write_pack_index_v1(f, entries, self.calculate_checksum())

    def create_index_v2(self, filename, progress=None, resolve_ext_ref=None):
        """Create a version 2 index file for this data file.

        Args:
          filename: Index filename.
          progress: Progress report function
        Returns: Checksum of index file
        """
        entries = self.sorted_entries(progress=progress, resolve_ext_ref=resolve_ext_ref)
        with GitFile(filename, 'wb') as f:
            return write_pack_index_v2(f, entries, self.calculate_checksum())

    def create_index(self, filename, progress=None, version=2, resolve_ext_ref=None):
        """Create an  index file for this data file.

        Args:
          filename: Index filename.
          progress: Progress report function
        Returns: Checksum of index file
        """
        if version == 1:
            return self.create_index_v1(filename, progress, resolve_ext_ref=resolve_ext_ref)
        elif version == 2:
            return self.create_index_v2(filename, progress, resolve_ext_ref=resolve_ext_ref)
        else:
            raise ValueError('unknown index format %d' % version)

    def get_stored_checksum(self):
        """Return the expected checksum stored in this pack."""
        self._file.seek(-20, SEEK_END)
        return self._file.read(20)

    def check(self):
        """Check the consistency of this pack."""
        actual = self.calculate_checksum()
        stored = self.get_stored_checksum()
        if actual != stored:
            raise ChecksumMismatch(stored, actual)

    def get_unpacked_object_at(self, offset: int, *, include_comp: bool=False) -> UnpackedObject:
        """Given offset in the packfile return a UnpackedObject."""
        assert offset >= self._header_size
        self._file.seek(offset)
        unpacked, _ = unpack_object(self._file.read, include_comp=include_comp)
        unpacked.offset = offset
        return unpacked

    def get_object_at(self, offset: int) -> Tuple[int, OldUnpackedObject]:
        """Given an offset in to the packfile return the object that is there.

        Using the associated index the location of an object can be looked up,
        and then the packfile can be asked directly for that object using this
        function.
        """
        try:
            return self._offset_cache[offset]
        except KeyError:
            pass
        unpacked = self.get_unpacked_object_at(offset, include_comp=False)
        return (unpacked.pack_type_num, unpacked._obj())