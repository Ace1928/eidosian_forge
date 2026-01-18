import struct
import time
import aiokafka.codec as codecs
from aiokafka.errors import CorruptRecordException, UnsupportedCodecError
from aiokafka.util import NO_EXTENSIONS
from aiokafka.codec import (
from .util import decode_varint, encode_varint, calc_crc32c, size_of_varint
class _DefaultRecordPy:
    __slots__ = ('_offset', '_timestamp', '_timestamp_type', '_key', '_value', '_headers')

    def __init__(self, offset, timestamp, timestamp_type, key, value, headers):
        self._offset = offset
        self._timestamp = timestamp
        self._timestamp_type = timestamp_type
        self._key = key
        self._value = value
        self._headers = headers

    @property
    def offset(self):
        return self._offset

    @property
    def timestamp(self):
        """ Epoch milliseconds
        """
        return self._timestamp

    @property
    def timestamp_type(self):
        """ CREATE_TIME(0) or APPEND_TIME(1)
        """
        return self._timestamp_type

    @property
    def key(self):
        """ Bytes key or None
        """
        return self._key

    @property
    def value(self):
        """ Bytes value or None
        """
        return self._value

    @property
    def headers(self):
        return self._headers

    @property
    def checksum(self):
        return None

    def __repr__(self):
        return f'DefaultRecord(offset={self._offset!r}, timestamp={self._timestamp!r}, timestamp_type={self._timestamp_type!r}, key={self._key!r}, value={self._value!r}, headers={self._headers!r})'