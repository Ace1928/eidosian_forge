from datetime import datetime as _DateTime
import sys
import struct
from .exceptions import BufferFull, OutOfData, ExtraData, FormatError, StackError
from .ext import ExtType, Timestamp
def _pack_map_pairs(self, n, pairs, nest_limit=DEFAULT_RECURSE_LIMIT):
    self._pack_map_header(n)
    for k, v in pairs:
        self._pack(k, nest_limit - 1)
        self._pack(v, nest_limit - 1)