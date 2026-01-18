import struct
from llvmlite.ir._utils import _StrCaching
def _as_half(value):
    """
    Truncate to half-precision float.
    """
    try:
        return struct.unpack('e', struct.pack('e', value))[0]
    except struct.error:
        return _as_float(value)