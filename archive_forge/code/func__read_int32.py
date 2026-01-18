import struct
import numpy as np
import tempfile
import zlib
import warnings
def _read_int32(f):
    """Read a signed 32-bit integer"""
    return np.int32(struct.unpack('>i', f.read(4))[0])