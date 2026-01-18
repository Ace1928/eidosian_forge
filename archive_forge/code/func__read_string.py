import struct
import numpy as np
import tempfile
import zlib
import warnings
def _read_string(f):
    """Read a string"""
    length = _read_long(f)
    if length > 0:
        chars = _read_bytes(f, length).decode('latin1')
        _align_32(f)
    else:
        chars = ''
    return chars