from .lib.py3compat import int2byte
from .lib import (BitStreamReader, BitStreamWriter, encode_bin,
from .core import (Struct, MetaField, StaticField, FormatField,
from .adapters import (BitIntegerAdapter, PaddingAdapter,
def Aligned(subcon, modulus=4, pattern=b'\x00'):
    """aligns subcon to modulus boundary using padding pattern
    * subcon - the subcon to align
    * modulus - the modulus boundary (default is 4)
    * pattern - the padding pattern (default is \\x00)
    """
    if modulus < 2:
        raise ValueError('modulus must be >= 2', modulus)

    def padlength(ctx):
        return (modulus - subcon._sizeof(ctx) % modulus) % modulus
    return SeqOfOne(subcon.name, subcon, Padding(padlength, pattern=pattern), nested=False)