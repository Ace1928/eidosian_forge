from datetime import datetime as _DateTime
import sys
import struct
from .exceptions import BufferFull, OutOfData, ExtraData, FormatError, StackError
from .ext import ExtType, Timestamp
def _pack_map_header(self, n):
    if n <= 15:
        return self._buffer.write(struct.pack('B', 128 + n))
    if n <= 65535:
        return self._buffer.write(struct.pack('>BH', 222, n))
    if n <= 4294967295:
        return self._buffer.write(struct.pack('>BI', 223, n))
    raise ValueError('Dict is too large')