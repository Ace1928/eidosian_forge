from .lib.py3compat import int2byte
from .lib import (BitStreamReader, BitStreamWriter, encode_bin,
from .core import (Struct, MetaField, StaticField, FormatField,
from .adapters import (BitIntegerAdapter, PaddingAdapter,
def PrefixedArray(subcon, length_field=UBInt8('length')):
    """an array prefixed by a length field.
    * subcon - the subcon to be repeated
    * length_field - a construct returning an integer
    """
    return LengthValueAdapter(Sequence(subcon.name, length_field, Array(lambda ctx: ctx[length_field.name], subcon), nested=False))