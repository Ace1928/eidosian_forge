import io
import sys
import numpy
import struct
import warnings
from enum import IntEnum
def _skip_unknown_chunk(fid, is_big_endian):
    if is_big_endian:
        fmt = '>I'
    else:
        fmt = '<I'
    data = fid.read(4)
    if data:
        size = struct.unpack(fmt, data)[0]
        fid.seek(size, 1)
        _handle_pad_byte(fid, size)