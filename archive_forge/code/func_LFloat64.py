from .lib.py3compat import int2byte
from .lib import (BitStreamReader, BitStreamWriter, encode_bin,
from .core import (Struct, MetaField, StaticField, FormatField,
from .adapters import (BitIntegerAdapter, PaddingAdapter,
def LFloat64(name):
    """little endian, 64-bit IEEE floating point number"""
    return FormatField(name, '<', 'd')