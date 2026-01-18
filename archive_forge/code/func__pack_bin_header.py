from datetime import datetime as _DateTime
import sys
import struct
from .exceptions import BufferFull, OutOfData, ExtraData, FormatError, StackError
from .ext import ExtType, Timestamp
def _pack_bin_header(self, n):
    if not self._use_bin_type:
        return self._pack_raw_header(n)
    elif n <= 255:
        return self._buffer.write(struct.pack('>BB', 196, n))
    elif n <= 65535:
        return self._buffer.write(struct.pack('>BH', 197, n))
    elif n <= 4294967295:
        return self._buffer.write(struct.pack('>BI', 198, n))
    else:
        raise ValueError('Bin is too large')