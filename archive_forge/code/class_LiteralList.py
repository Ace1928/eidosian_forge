from collections.abc import Iterable
from collections.abc import Sequence as pySequence
from types import MappingProxyType
from .abstract import (
from .common import (
from .misc import Undefined, unliteral, Optional, NoneType
from ..typeconv import Conversion
from ..errors import TypingError
from .. import utils
class LiteralList(Literal, ConstSized, Hashable):
    """A heterogeneous immutable list (basically a tuple with list semantics).
    """
    mutable = False

    def __init__(self, literal_value):
        self.is_types_iterable(literal_value)
        self._literal_init(list(literal_value))
        self.types = tuple(literal_value)
        self.count = len(self.types)
        self.name = 'LiteralList({})'.format(literal_value)

    def __getitem__(self, i):
        """
        Return element at position i
        """
        return self.types[i]

    def __len__(self):
        return len(self.types)

    def __iter__(self):
        return iter(self.types)

    @classmethod
    def from_types(cls, tys):
        return LiteralList(tys)

    @staticmethod
    def is_types_iterable(types):
        if not isinstance(types, Iterable):
            raise TypingError("Argument 'types' is not iterable")

    @property
    def iterator_type(self):
        return ListIter(self)

    def __unliteral__(self):
        return Poison(self)

    def unify(self, typingctx, other):
        """
        Unify this with the *other* one.
        """
        if isinstance(other, LiteralList) and self.count == other.count:
            tys = []
            for i1, i2 in zip(self.types, other.types):
                tys.append(typingctx.unify_pairs(i1, i2))
            if all(tys):
                return LiteralList(tys)