from struct import Struct as Packer
from .lib.py3compat import BytesIO, advance_iterator, bchr
from .lib import Container, ListContainer, LazyContainer
class StaticField(Construct):
    """
    A fixed-size byte field.

    :param str name: field name
    :param int length: number of bytes in the field
    """
    __slots__ = ['length']

    def __init__(self, name, length):
        Construct.__init__(self, name)
        self.length = length

    def _parse(self, stream, context):
        return _read_stream(stream, self.length)

    def _build(self, obj, stream, context):
        _write_stream(stream, self.length, obj)

    def _sizeof(self, context):
        return self.length