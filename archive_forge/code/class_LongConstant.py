import sys
from OpenGL._bytes import bytes,unicode,as_8_bit, long, integer_types, maxsize
from OpenGL import _configflags
class LongConstant(NumericConstant, long):
    """Long integer constant"""