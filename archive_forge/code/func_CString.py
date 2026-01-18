from .lib.py3compat import int2byte
from .lib import (BitStreamReader, BitStreamWriter, encode_bin,
from .core import (Struct, MetaField, StaticField, FormatField,
from .adapters import (BitIntegerAdapter, PaddingAdapter,
def CString(name, terminators=b'\x00', encoding=None, char_field=Field(None, 1)):
    """
    A string ending in a terminator.

    ``CString`` is similar to the strings of C, C++, and other related
    programming languages.

    By default, the terminator is the NULL byte (b``0x00``).

    :param str name: name
    :param iterable terminators: sequence of valid terminators, in order of
                                 preference
    :param str encoding: encoding (e.g. "utf8") or None for no encoding
    :param ``Construct`` char_field: construct representing a single character

    >>> foo = CString("foo")
    >>> foo.parse(b"hello\\x00")
    b'hello'
    >>> foo.build(b"hello")
    b'hello\\x00'
    >>> foo = CString("foo", terminators = b"XYZ")
    >>> foo.parse(b"helloX")
    b'hello'
    >>> foo.parse(b"helloY")
    b'hello'
    >>> foo.parse(b"helloZ")
    b'hello'
    >>> foo.build(b"hello")
    b'helloX'
    """
    return Rename(name, CStringAdapter(RepeatUntil(lambda obj, ctx: obj in terminators, char_field), terminators=terminators, encoding=encoding))