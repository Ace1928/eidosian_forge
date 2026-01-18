from .lib.py3compat import int2byte
from .lib import (BitStreamReader, BitStreamWriter, encode_bin,
from .core import (Struct, MetaField, StaticField, FormatField,
from .adapters import (BitIntegerAdapter, PaddingAdapter,
def AlignedStruct(name, *subcons, **kw):
    """a struct of aligned fields
    * name - the name of the struct
    * subcons - the subcons that make up this structure
    * kw - keyword arguments to pass to Aligned: 'modulus' and 'pattern'
    """
    return Struct(name, *(Aligned(sc, **kw) for sc in subcons))