from .lib.py3compat import int2byte
from .lib import (BitStreamReader, BitStreamWriter, encode_bin,
from .core import (Struct, MetaField, StaticField, FormatField,
from .adapters import (BitIntegerAdapter, PaddingAdapter,
def PascalString(name, length_field=UBInt8('length'), encoding=None):
    """
    A length-prefixed string.

    ``PascalString`` is named after the string types of Pascal, which are
    length-prefixed. Lisp strings also follow this convention.

    The length field will appear in the same ``Container`` as the
    ``PascalString``, with the given name.

    :param str name: name
    :param ``Construct`` length_field: a field which will store the length of
                                       the string
    :param str encoding: encoding (e.g. "utf8") or None for no encoding

    >>> foo = PascalString("foo")
    >>> foo.parse("\\x05hello")
    'hello'
    >>> foo.build("hello world")
    '\\x0bhello world'
    >>>
    >>> foo = PascalString("foo", length_field = UBInt16("length"))
    >>> foo.parse("\\x00\\x05hello")
    'hello'
    >>> foo.build("hello")
    '\\x00\\x05hello'
    """
    return StringAdapter(LengthValueAdapter(Sequence(name, length_field, Field('data', lambda ctx: ctx[length_field.name]))), encoding=encoding)