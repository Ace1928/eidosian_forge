import operator
from numba.core import types
from .templates import (ConcreteTemplate, AbstractTemplate, AttributeTemplate,
from numba.core.typing import collections
def _resolve_operator(self, set, args, kws):
    assert not kws
    iterable, = args
    if isinstance(iterable, types.Set) and iterable.dtype == set.dtype:
        return signature(set, iterable)