import struct
import numpy as np
import tempfile
import zlib
import warnings
def _read_string_data(f):
    """Read a data string (length is specified twice)"""
    length = _read_long(f)
    if length > 0:
        length = _read_long(f)
        string_data = _read_bytes(f, length)
        _align_32(f)
    else:
        string_data = ''
    return string_data