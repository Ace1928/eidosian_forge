import sys
import h5py
import numpy as np
from . import core
def _check_return_dtype_endianess(endian='native'):
    little_endian = sys.byteorder == 'little'
    endianess = '='
    if endian == 'little':
        endianess = little_endian and endianess or '<'
    elif endian == 'big':
        endianess = not little_endian and endianess or '>'
    elif endian == 'native':
        pass
    else:
        raise ValueError("'endian' keyword argument must be 'little','big' or 'native', got '%s'" % endian)
    return endianess