import struct
import numpy as np
import tempfile
import zlib
import warnings
def _read_uint64(f):
    """Read an unsigned 64-bit integer"""
    return np.uint64(struct.unpack('>Q', f.read(8))[0])