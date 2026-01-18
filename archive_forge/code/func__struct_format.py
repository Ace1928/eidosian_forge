import math
import struct
from ctypes import create_string_buffer
def _struct_format(size, signed):
    if size == 1:
        return 'b' if signed else 'B'
    elif size == 2:
        return 'h' if signed else 'H'
    elif size == 4:
        return 'i' if signed else 'I'