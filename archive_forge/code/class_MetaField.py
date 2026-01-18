from struct import Struct as Packer
from .lib.py3compat import BytesIO, advance_iterator, bchr
from .lib import Container, ListContainer, LazyContainer
class MetaField(Construct):
    """
    A variable-length field. The length is obtained at runtime from a
    function.

    :param str name: name of the field
    :param callable lengthfunc: callable that takes a context and returns
                                length as an int

    >>> foo = Struct("foo",
    ...     Byte("length"),
    ...     MetaField("data", lambda ctx: ctx["length"])
    ... )
    >>> foo.parse("\\x03ABC")
    Container(data = 'ABC', length = 3)
    >>> foo.parse("\\x04ABCD")
    Container(data = 'ABCD', length = 4)
    """
    __slots__ = ['lengthfunc']

    def __init__(self, name, lengthfunc):
        Construct.__init__(self, name)
        self.lengthfunc = lengthfunc
        self._set_flag(self.FLAG_DYNAMIC)

    def _parse(self, stream, context):
        return _read_stream(stream, self.lengthfunc(context))

    def _build(self, obj, stream, context):
        _write_stream(stream, self.lengthfunc(context), obj)

    def _sizeof(self, context):
        return self.lengthfunc(context)