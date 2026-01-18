from collections.abc import Iterable
from collections.abc import Sequence as pySequence
from types import MappingProxyType
from .abstract import (
from .common import (
from .misc import Undefined, unliteral, Optional, NoneType
from ..typeconv import Conversion
from ..errors import TypingError
from .. import utils
class NamedUniTuple(_HomogeneousTuple, BaseNamedTuple):

    def __init__(self, dtype, count, cls):
        self.dtype = dtype
        self.count = count
        self.fields = tuple(cls._fields)
        self.instance_class = cls
        name = '%s(%s x %d)' % (cls.__name__, dtype, count)
        super(NamedUniTuple, self).__init__(name)

    @property
    def iterator_type(self):
        return UniTupleIter(self)

    @property
    def key(self):
        return (self.instance_class, self.dtype, self.count)