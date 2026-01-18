from .lib.py3compat import int2byte
from .lib import (BitStreamReader, BitStreamWriter, encode_bin,
from .core import (Struct, MetaField, StaticField, FormatField,
from .adapters import (BitIntegerAdapter, PaddingAdapter,
def NFloat64(name):
    """native endianity, 64-bit IEEE floating point number"""
    return FormatField(name, '=', 'd')