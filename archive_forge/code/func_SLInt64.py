from .lib.py3compat import int2byte
from .lib import (BitStreamReader, BitStreamWriter, encode_bin,
from .core import (Struct, MetaField, StaticField, FormatField,
from .adapters import (BitIntegerAdapter, PaddingAdapter,
def SLInt64(name):
    """signed, little endian 64-bit integer"""
    return FormatField(name, '<', 'q')