import struct
import numpy as np
import tempfile
import zlib
import warnings
def _read_uint16(f):
    """Read an unsigned 16-bit integer"""
    return np.uint16(struct.unpack('>H', f.read(4)[2:4])[0])