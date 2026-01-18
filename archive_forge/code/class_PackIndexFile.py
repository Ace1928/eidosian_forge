import zlib
from gitdb.exc import (
from gitdb.util import (
from gitdb.fun import (
from gitdb.base import (      # Amazing !
from gitdb.stream import (
from struct import pack
from binascii import crc32
from gitdb.const import NULL_BYTE
import tempfile
import array
import os
import sys
class PackIndexFile(LazyMixin):
    """A pack index provides offsets into the corresponding pack, allowing to find
    locations for offsets faster."""
    _sha_list_offset = 8 + 1024
    index_v2_signature = b'\xfftOc'
    index_version_default = 2

    def __init__(self, indexpath):
        super().__init__()
        self._indexpath = indexpath

    def close(self):
        mman.force_map_handle_removal_win(self._indexpath)
        self._cursor = None

    def _set_cache_(self, attr):
        if attr == '_packfile_checksum':
            self._packfile_checksum = self._cursor.map()[-40:-20]
        elif attr == '_packfile_checksum':
            self._packfile_checksum = self._cursor.map()[-20:]
        elif attr == '_cursor':
            self._cursor = mman.make_cursor(self._indexpath).use_region()
            if mman.window_size() > 0 and self._cursor.file_size() > mman.window_size():
                raise AssertionError('The index file at %s is too large to fit into a mapped window (%i > %i). This is a limitation of the implementation' % (self._indexpath, self._cursor.file_size(), mman.window_size()))
        else:
            mmap = self._cursor.map()
            self._version = mmap[:4] == self.index_v2_signature and 2 or 1
            if self._version == 2:
                version_id = unpack_from('>L', mmap, 4)[0]
                assert version_id == self._version, 'Unsupported index version: %i' % version_id
            for fname in ('entry', 'offset', 'sha', 'crc'):
                setattr(self, fname, getattr(self, '_%s_v%i' % (fname, self._version)))
            self._initialize()

    def _entry_v1(self, i):
        """:return: tuple(offset, binsha, 0)"""
        return unpack_from('>L20s', self._cursor.map(), 1024 + i * 24) + (0,)

    def _offset_v1(self, i):
        """see ``_offset_v2``"""
        return unpack_from('>L', self._cursor.map(), 1024 + i * 24)[0]

    def _sha_v1(self, i):
        """see ``_sha_v2``"""
        base = 1024 + i * 24 + 4
        return self._cursor.map()[base:base + 20]

    def _crc_v1(self, i):
        """unsupported"""
        return 0

    def _entry_v2(self, i):
        """:return: tuple(offset, binsha, crc)"""
        return (self._offset_v2(i), self._sha_v2(i), self._crc_v2(i))

    def _offset_v2(self, i):
        """:return: 32 or 64 byte offset into pack files. 64 byte offsets will only
            be returned if the pack is larger than 4 GiB, or 2^32"""
        offset = unpack_from('>L', self._cursor.map(), self._pack_offset + i * 4)[0]
        if offset & 2147483648:
            offset = unpack_from('>Q', self._cursor.map(), self._pack_64_offset + (offset & ~2147483648) * 8)[0]
        return offset

    def _sha_v2(self, i):
        """:return: sha at the given index of this file index instance"""
        base = self._sha_list_offset + i * 20
        return self._cursor.map()[base:base + 20]

    def _crc_v2(self, i):
        """:return: 4 bytes crc for the object at index i"""
        return unpack_from('>L', self._cursor.map(), self._crc_list_offset + i * 4)[0]

    def _initialize(self):
        """initialize base data"""
        self._fanout_table = self._read_fanout((self._version == 2) * 8)
        if self._version == 2:
            self._crc_list_offset = self._sha_list_offset + self.size() * 20
            self._pack_offset = self._crc_list_offset + self.size() * 4
            self._pack_64_offset = self._pack_offset + self.size() * 4

    def _read_fanout(self, byte_offset):
        """Generate a fanout table from our data"""
        d = self._cursor.map()
        out = list()
        append = out.append
        for i in range(256):
            append(unpack_from('>L', d, byte_offset + i * 4)[0])
        return out

    def version(self):
        return self._version

    def size(self):
        """:return: amount of objects referred to by this index"""
        return self._fanout_table[255]

    def path(self):
        """:return: path to the packindexfile"""
        return self._indexpath

    def packfile_checksum(self):
        """:return: 20 byte sha representing the sha1 hash of the pack file"""
        return self._cursor.map()[-40:-20]

    def indexfile_checksum(self):
        """:return: 20 byte sha representing the sha1 hash of this index file"""
        return self._cursor.map()[-20:]

    def offsets(self):
        """:return: sequence of all offsets in the order in which they were written

        **Note:** return value can be random accessed, but may be immmutable"""
        if self._version == 2:
            a = array.array('I')
            a.frombytes(self._cursor.map()[self._pack_offset:self._pack_64_offset])
            if sys.byteorder == 'little':
                a.byteswap()
            return a
        else:
            return tuple((self.offset(index) for index in range(self.size())))

    def sha_to_index(self, sha):
        """
        :return: index usable with the ``offset`` or ``entry`` method, or None
            if the sha was not found in this pack index
        :param sha: 20 byte sha to lookup"""
        first_byte = byte_ord(sha[0])
        get_sha = self.sha
        lo = 0
        if first_byte != 0:
            lo = self._fanout_table[first_byte - 1]
        hi = self._fanout_table[first_byte]
        while lo < hi:
            mid = (lo + hi) // 2
            mid_sha = get_sha(mid)
            if sha < mid_sha:
                hi = mid
            elif sha == mid_sha:
                return mid
            else:
                lo = mid + 1
        return None

    def partial_sha_to_index(self, partial_bin_sha, canonical_length):
        """
        :return: index as in `sha_to_index` or None if the sha was not found in this
            index file
        :param partial_bin_sha: an at least two bytes of a partial binary sha as bytes
        :param canonical_length: length of the original hexadecimal representation of the
            given partial binary sha
        :raise AmbiguousObjectName:"""
        if len(partial_bin_sha) < 2:
            raise ValueError('Require at least 2 bytes of partial sha')
        assert isinstance(partial_bin_sha, bytes), 'partial_bin_sha must be bytes'
        first_byte = byte_ord(partial_bin_sha[0])
        get_sha = self.sha
        lo = 0
        if first_byte != 0:
            lo = self._fanout_table[first_byte - 1]
        hi = self._fanout_table[first_byte]
        filled_sha = partial_bin_sha + NULL_BYTE * (20 - len(partial_bin_sha))
        while lo < hi:
            mid = (lo + hi) // 2
            mid_sha = get_sha(mid)
            if filled_sha < mid_sha:
                hi = mid
            elif filled_sha == mid_sha:
                lo = mid
                break
            else:
                lo = mid + 1
        if lo < self.size():
            cur_sha = get_sha(lo)
            if is_equal_canonical_sha(canonical_length, partial_bin_sha, cur_sha):
                next_sha = None
                if lo + 1 < self.size():
                    next_sha = get_sha(lo + 1)
                if next_sha and next_sha == cur_sha:
                    raise AmbiguousObjectName(partial_bin_sha)
                return lo
        return None
    if 'PackIndexFile_sha_to_index' in globals():

        def sha_to_index(self, sha):
            return PackIndexFile_sha_to_index(self, sha)