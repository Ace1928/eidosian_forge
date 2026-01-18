import math
import struct
from ctypes import create_string_buffer
def _overflow(val, size, signed=True):
    minval = _get_minval(size, signed)
    maxval = _get_maxval(size, signed)
    if minval <= val <= maxval:
        return val
    bits = size * 8
    if signed:
        offset = 2 ** (bits - 1)
        return (val + offset) % 2 ** bits - offset
    else:
        return val % 2 ** bits