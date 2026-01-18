from datetime import datetime as _DateTime
import sys
import struct
from .exceptions import BufferFull, OutOfData, ExtraData, FormatError, StackError
from .ext import ExtType, Timestamp
def _get_data_from_buffer(obj):
    view = memoryview(obj)
    if view.itemsize != 1:
        raise ValueError('cannot unpack from multi-byte object')
    return view