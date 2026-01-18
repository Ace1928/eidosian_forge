from .lib.py3compat import int2byte
from .lib import (BitStreamReader, BitStreamWriter, encode_bin,
from .core import (Struct, MetaField, StaticField, FormatField,
from .adapters import (BitIntegerAdapter, PaddingAdapter,
def EmbeddedBitStruct(*subcons):
    """an embedded BitStruct. no name is necessary.
    * subcons - the subcons that make up this structure
    """
    return Bitwise(Embedded(Struct(None, *subcons)))