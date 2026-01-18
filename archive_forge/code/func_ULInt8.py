from .lib.py3compat import int2byte
from .lib import (BitStreamReader, BitStreamWriter, encode_bin,
from .core import (Struct, MetaField, StaticField, FormatField,
from .adapters import (BitIntegerAdapter, PaddingAdapter,
def ULInt8(name):
    """unsigned, little endian 8-bit integer"""
    return FormatField(name, '<', 'B')