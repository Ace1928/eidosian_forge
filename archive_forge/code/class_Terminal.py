from typing import Optional, Tuple, ClassVar, Sequence
from .utils import Serialize
class Terminal(Symbol):
    __serialize_fields__ = ('name', 'filter_out')
    is_term: ClassVar[bool] = True

    def __init__(self, name, filter_out=False):
        self.name = name
        self.filter_out = filter_out

    @property
    def fullrepr(self):
        return '%s(%r, %r)' % (type(self).__name__, self.name, self.filter_out)

    def renamed(self, f):
        return type(self)(f(self.name), self.filter_out)