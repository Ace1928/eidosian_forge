from .lib.py3compat import int2byte
from .lib import (BitStreamReader, BitStreamWriter, encode_bin,
from .core import (Struct, MetaField, StaticField, FormatField,
from .adapters import (BitIntegerAdapter, PaddingAdapter,
def SLInt16(name):
    """signed, little endian 16-bit integer"""
    return FormatField(name, '<', 'h')