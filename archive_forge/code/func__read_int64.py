import struct
import numpy as np
import tempfile
import zlib
import warnings
def _read_int64(f):
    """Read a signed 64-bit integer"""
    return np.int64(struct.unpack('>q', f.read(8))[0])