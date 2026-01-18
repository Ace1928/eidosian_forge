import binascii
import os
import posixpath
import stat
import warnings
import zlib
from collections import namedtuple
from hashlib import sha1
from io import BytesIO
from typing import (
from .errors import (
from .file import GitFile
class ShaFile:
    """A git SHA file."""
    __slots__ = ('_chunked_text', '_sha', '_needs_serialization')
    _needs_serialization: bool
    type_name: bytes
    type_num: int
    _chunked_text: Optional[List[bytes]]
    _sha: Union[FixedSha, None, 'HASH']

    @staticmethod
    def _parse_legacy_object_header(magic, f: BinaryIO) -> 'ShaFile':
        """Parse a legacy object, creating it but not reading the file."""
        bufsize = 1024
        decomp = zlib.decompressobj()
        header = decomp.decompress(magic)
        start = 0
        end = -1
        while end < 0:
            extra = f.read(bufsize)
            header += decomp.decompress(extra)
            magic += extra
            end = header.find(b'\x00', start)
            start = len(header)
        header = header[:end]
        type_name, size = header.split(b' ', 1)
        try:
            int(size)
        except ValueError as exc:
            raise ObjectFormatException('Object size not an integer: %s' % exc) from exc
        obj_class = object_class(type_name)
        if not obj_class:
            raise ObjectFormatException('Not a known type: %s' % type_name.decode('ascii'))
        return obj_class()

    def _parse_legacy_object(self, map) -> None:
        """Parse a legacy object, setting the raw string."""
        text = _decompress(map)
        header_end = text.find(b'\x00')
        if header_end < 0:
            raise ObjectFormatException('Invalid object header, no \\0')
        self.set_raw_string(text[header_end + 1:])

    def as_legacy_object_chunks(self, compression_level: int=-1) -> Iterator[bytes]:
        """Return chunks representing the object in the experimental format.

        Returns: List of strings
        """
        compobj = zlib.compressobj(compression_level)
        yield compobj.compress(self._header())
        for chunk in self.as_raw_chunks():
            yield compobj.compress(chunk)
        yield compobj.flush()

    def as_legacy_object(self, compression_level: int=-1) -> bytes:
        """Return string representing the object in the experimental format."""
        return b''.join(self.as_legacy_object_chunks(compression_level=compression_level))

    def as_raw_chunks(self) -> List[bytes]:
        """Return chunks with serialization of the object.

        Returns: List of strings, not necessarily one per line
        """
        if self._needs_serialization:
            self._sha = None
            self._chunked_text = self._serialize()
            self._needs_serialization = False
        return self._chunked_text

    def as_raw_string(self) -> bytes:
        """Return raw string with serialization of the object.

        Returns: String object
        """
        return b''.join(self.as_raw_chunks())

    def __bytes__(self) -> bytes:
        """Return raw string serialization of this object."""
        return self.as_raw_string()

    def __hash__(self):
        """Return unique hash for this object."""
        return hash(self.id)

    def as_pretty_string(self) -> bytes:
        """Return a string representing this object, fit for display."""
        return self.as_raw_string()

    def set_raw_string(self, text: bytes, sha: Optional[ObjectID]=None) -> None:
        """Set the contents of this object from a serialized string."""
        if not isinstance(text, bytes):
            raise TypeError('Expected bytes for text, got %r' % text)
        self.set_raw_chunks([text], sha)

    def set_raw_chunks(self, chunks: List[bytes], sha: Optional[ObjectID]=None) -> None:
        """Set the contents of this object from a list of chunks."""
        self._chunked_text = chunks
        self._deserialize(chunks)
        if sha is None:
            self._sha = None
        else:
            self._sha = FixedSha(sha)
        self._needs_serialization = False

    @staticmethod
    def _parse_object_header(magic, f):
        """Parse a new style object, creating it but not reading the file."""
        num_type = ord(magic[0:1]) >> 4 & 7
        obj_class = object_class(num_type)
        if not obj_class:
            raise ObjectFormatException('Not a known type %d' % num_type)
        return obj_class()

    def _parse_object(self, map) -> None:
        """Parse a new style object, setting self._text."""
        byte = ord(map[0:1])
        used = 1
        while byte & 128 != 0:
            byte = ord(map[used:used + 1])
            used += 1
        raw = map[used:]
        self.set_raw_string(_decompress(raw))

    @classmethod
    def _is_legacy_object(cls, magic: bytes) -> bool:
        b0 = ord(magic[0:1])
        b1 = ord(magic[1:2])
        word = (b0 << 8) + b1
        return b0 & 143 == 8 and word % 31 == 0

    @classmethod
    def _parse_file(cls, f):
        map = f.read()
        if not map:
            raise EmptyFileException('Corrupted empty file detected')
        if cls._is_legacy_object(map):
            obj = cls._parse_legacy_object_header(map, f)
            obj._parse_legacy_object(map)
        else:
            obj = cls._parse_object_header(map, f)
            obj._parse_object(map)
        return obj

    def __init__(self) -> None:
        """Don't call this directly."""
        self._sha = None
        self._chunked_text = []
        self._needs_serialization = True

    def _deserialize(self, chunks: List[bytes]) -> None:
        raise NotImplementedError(self._deserialize)

    def _serialize(self) -> List[bytes]:
        raise NotImplementedError(self._serialize)

    @classmethod
    def from_path(cls, path):
        """Open a SHA file from disk."""
        with GitFile(path, 'rb') as f:
            return cls.from_file(f)

    @classmethod
    def from_file(cls, f):
        """Get the contents of a SHA file on disk."""
        try:
            obj = cls._parse_file(f)
            obj._sha = None
            return obj
        except (IndexError, ValueError) as exc:
            raise ObjectFormatException('invalid object header') from exc

    @staticmethod
    def from_raw_string(type_num, string, sha=None):
        """Creates an object of the indicated type from the raw string given.

        Args:
          type_num: The numeric type of the object.
          string: The raw uncompressed contents.
          sha: Optional known sha for the object
        """
        cls = object_class(type_num)
        if cls is None:
            raise AssertionError('unsupported class type num: %d' % type_num)
        obj = cls()
        obj.set_raw_string(string, sha)
        return obj

    @staticmethod
    def from_raw_chunks(type_num: int, chunks: List[bytes], sha: Optional[ObjectID]=None):
        """Creates an object of the indicated type from the raw chunks given.

        Args:
          type_num: The numeric type of the object.
          chunks: An iterable of the raw uncompressed contents.
          sha: Optional known sha for the object
        """
        cls = object_class(type_num)
        if cls is None:
            raise AssertionError('unsupported class type num: %d' % type_num)
        obj = cls()
        obj.set_raw_chunks(chunks, sha)
        return obj

    @classmethod
    def from_string(cls, string):
        """Create a ShaFile from a string."""
        obj = cls()
        obj.set_raw_string(string)
        return obj

    def _check_has_member(self, member, error_msg):
        """Check that the object has a given member variable.

        Args:
          member: the member variable to check for
          error_msg: the message for an error if the member is missing
        Raises:
          ObjectFormatException: with the given error_msg if member is
            missing or is None
        """
        if getattr(self, member, None) is None:
            raise ObjectFormatException(error_msg)

    def check(self) -> None:
        """Check this object for internal consistency.

        Raises:
          ObjectFormatException: if the object is malformed in some way
          ChecksumMismatch: if the object was created with a SHA that does
            not match its contents
        """
        old_sha = self.id
        try:
            self._deserialize(self.as_raw_chunks())
            self._sha = None
            new_sha = self.id
        except Exception as exc:
            raise ObjectFormatException(exc) from exc
        if old_sha != new_sha:
            raise ChecksumMismatch(new_sha, old_sha)

    def _header(self):
        return object_header(self.type_num, self.raw_length())

    def raw_length(self) -> int:
        """Returns the length of the raw string of this object."""
        return sum(map(len, self.as_raw_chunks()))

    def sha(self):
        """The SHA1 object that is the name of this object."""
        if self._sha is None or self._needs_serialization:
            new_sha = sha1()
            new_sha.update(self._header())
            for chunk in self.as_raw_chunks():
                new_sha.update(chunk)
            self._sha = new_sha
        return self._sha

    def copy(self):
        """Create a new copy of this SHA1 object from its raw string."""
        obj_class = object_class(self.type_num)
        if obj_class is None:
            raise AssertionError('invalid type num %d' % self.type_num)
        return obj_class.from_raw_string(self.type_num, self.as_raw_string(), self.id)

    @property
    def id(self):
        """The hex SHA of this object."""
        return self.sha().hexdigest().encode('ascii')

    def __repr__(self) -> str:
        return f'<{self.__class__.__name__} {self.id}>'

    def __ne__(self, other):
        """Check whether this object does not match the other."""
        return not isinstance(other, ShaFile) or self.id != other.id

    def __eq__(self, other):
        """Return True if the SHAs of the two objects match."""
        return isinstance(other, ShaFile) and self.id == other.id

    def __lt__(self, other):
        """Return whether SHA of this object is less than the other."""
        if not isinstance(other, ShaFile):
            raise TypeError
        return self.id < other.id

    def __le__(self, other):
        """Check whether SHA of this object is less than or equal to the other."""
        if not isinstance(other, ShaFile):
            raise TypeError
        return self.id <= other.id