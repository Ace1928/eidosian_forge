import struct
import time
from binascii import crc32
import aiokafka.codec as codecs
from aiokafka.codec import (
from aiokafka.errors import CorruptRecordException, UnsupportedCodecError
from aiokafka.util import NO_EXTENSIONS
class _LegacyRecordMetadataPy:
    __slots__ = ('_crc', '_size', '_timestamp', '_offset')

    def __init__(self, offset, crc, size, timestamp):
        self._offset = offset
        self._crc = crc
        self._size = size
        self._timestamp = timestamp

    @property
    def offset(self):
        return self._offset

    @property
    def crc(self):
        return self._crc

    @property
    def size(self):
        return self._size

    @property
    def timestamp(self):
        return self._timestamp

    def __repr__(self):
        return f'LegacyRecordMetadata(offset={self._offset!r}, crc={self._crc!r}, size={self._size!r}, timestamp={self._timestamp!r})'