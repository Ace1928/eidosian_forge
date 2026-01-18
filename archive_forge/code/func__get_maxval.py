import math
import struct
from ctypes import create_string_buffer
def _get_maxval(size, signed=True):
    if signed and size == 1:
        return 127
    elif size == 1:
        return 255
    elif signed and size == 2:
        return 32767
    elif size == 2:
        return 65535
    elif signed and size == 4:
        return 2147483647
    elif size == 4:
        return 4294967295