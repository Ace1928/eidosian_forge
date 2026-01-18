import math
import struct
from ctypes import create_string_buffer
def _put_sample(cp, size, i, val, signed=True):
    fmt = _struct_format(size, signed)
    struct.pack_into(fmt, cp, i * size, val)