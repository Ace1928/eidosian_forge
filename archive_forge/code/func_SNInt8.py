from .lib.py3compat import int2byte
from .lib import (BitStreamReader, BitStreamWriter, encode_bin,
from .core import (Struct, MetaField, StaticField, FormatField,
from .adapters import (BitIntegerAdapter, PaddingAdapter,
def SNInt8(name):
    """signed, native endianity 8-bit integer"""
    return FormatField(name, '=', 'b')