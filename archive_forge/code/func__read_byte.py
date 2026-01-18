import struct
import numpy as np
import tempfile
import zlib
import warnings
def _read_byte(f):
    """Read a single byte"""
    return np.uint8(struct.unpack('>B', f.read(4)[:1])[0])