from .lib.py3compat import int2byte
from .lib import (BitStreamReader, BitStreamWriter, encode_bin,
from .core import (Struct, MetaField, StaticField, FormatField,
from .adapters import (BitIntegerAdapter, PaddingAdapter,
def String(name, length, encoding=None, padchar=None, paddir='right', trimdir='right'):
    """
    A configurable, fixed-length string field.

    The padding character must be specified for padding and trimming to work.

    :param str name: name
    :param int length: length, in bytes
    :param str encoding: encoding (e.g. "utf8") or None for no encoding
    :param str padchar: optional character to pad out strings
    :param str paddir: direction to pad out strings; one of "right", "left",
                       or "both"
    :param str trim: direction to trim strings; one of "right", "left"

    >>> from construct import String
    >>> String("foo", 5).parse("hello")
    'hello'
    >>>
    >>> String("foo", 12, encoding = "utf8").parse("hello joh\\xd4\\x83n")
    u'hello joh\\u0503n'
    >>>
    >>> foo = String("foo", 10, padchar = "X", paddir = "right")
    >>> foo.parse("helloXXXXX")
    'hello'
    >>> foo.build("hello")
    'helloXXXXX'
    """
    con = StringAdapter(Field(name, length), encoding=encoding)
    if padchar is not None:
        con = PaddedStringAdapter(con, padchar=padchar, paddir=paddir, trimdir=trimdir)
    return con