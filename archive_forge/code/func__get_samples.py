import math
import struct
from ctypes import create_string_buffer
def _get_samples(cp, size, signed=True):
    for i in range(_sample_count(cp, size)):
        yield _get_sample(cp, size, i, signed)