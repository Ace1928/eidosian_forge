from .lib.py3compat import int2byte
from .lib import (BitStreamReader, BitStreamWriter, encode_bin,
from .core import (Struct, MetaField, StaticField, FormatField,
from .adapters import (BitIntegerAdapter, PaddingAdapter,
def BitField(name, length, swapped=False, signed=False, bytesize=8):
    """
    BitFields, as the name suggests, are fields that operate on raw, unaligned
    bits, and therefore must be enclosed in a BitStruct. Using them is very
    similar to all normal fields: they take a name and a length (in bits).

    :param str name: name of the field
    :param int length: number of bits in the field, or a function that takes
                       the context as its argument and returns the length
    :param bool swapped: whether the value is byte-swapped
    :param bool signed: whether the value is signed
    :param int bytesize: number of bits per byte, for byte-swapping

    >>> foo = BitStruct("foo",
    ...     BitField("a", 3),
    ...     Flag("b"),
    ...     Padding(3),
    ...     Nibble("c"),
    ...     BitField("d", 5),
    ... )
    >>> foo.parse("\\xe1\\x1f")
    Container(a = 7, b = False, c = 8, d = 31)
    >>> foo = BitStruct("foo",
    ...     BitField("a", 3),
    ...     Flag("b"),
    ...     Padding(3),
    ...     Nibble("c"),
    ...     Struct("bar",
    ...             Nibble("d"),
    ...             Bit("e"),
    ...     )
    ... )
    >>> foo.parse("\\xe1\\x1f")
    Container(a = 7, b = False, bar = Container(d = 15, e = 1), c = 8)
    """
    return BitIntegerAdapter(Field(name, length), length, swapped=swapped, signed=signed, bytesize=bytesize)