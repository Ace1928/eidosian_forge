import struct
from llvmlite.ir._utils import _StrCaching
class DoubleType(_BaseFloatType):
    """
    The type for double-precision floats.
    """
    null = '0.0'
    intrinsic_name = 'f64'

    def __str__(self):
        return 'double'

    def format_constant(self, value):
        return _format_double(value)