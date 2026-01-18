from collections.abc import Sequence, Hashable
from numbers import Integral
from functools import reduce
from typing import Generic, TypeVar
class _EmptyPList(_PListBase):
    __slots__ = ()

    def __bool__(self):
        return False
    __nonzero__ = __bool__

    @property
    def first(self):
        raise AttributeError('Empty PList has no first')

    @property
    def rest(self):
        return self