from struct import Struct as Packer
from .lib.py3compat import BytesIO, advance_iterator, bchr
from .lib import Container, ListContainer, LazyContainer
class LazyBound(Construct):
    """
    Lazily bound construct, useful for constructs that need to make cyclic
    references (linked-lists, expression trees, etc.).

    Parameters:


    Example:
    foo = Struct("foo",
        UBInt8("bar"),
        LazyBound("next", lambda: foo),
    )
    """
    __slots__ = ['bindfunc', 'bound']

    def __init__(self, name, bindfunc):
        Construct.__init__(self, name)
        self.bound = None
        self.bindfunc = bindfunc

    def _parse(self, stream, context):
        if self.bound is None:
            self.bound = self.bindfunc()
        return self.bound._parse(stream, context)

    def _build(self, obj, stream, context):
        if self.bound is None:
            self.bound = self.bindfunc()
        self.bound._build(obj, stream, context)

    def _sizeof(self, context):
        if self.bound is None:
            self.bound = self.bindfunc()
        return self.bound._sizeof(context)