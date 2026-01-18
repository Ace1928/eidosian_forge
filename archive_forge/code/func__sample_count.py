import math
import struct
from ctypes import create_string_buffer
def _sample_count(cp, size):
    return len(cp) / size