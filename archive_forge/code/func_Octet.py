from .lib.py3compat import int2byte
from .lib import (BitStreamReader, BitStreamWriter, encode_bin,
from .core import (Struct, MetaField, StaticField, FormatField,
from .adapters import (BitIntegerAdapter, PaddingAdapter,
def Octet(name):
    """an 8-bit BitField; must be enclosed in a BitStruct"""
    return BitField(name, 8)