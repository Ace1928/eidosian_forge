import operator
from numba.core import types
from .templates import (ConcreteTemplate, AbstractTemplate, AttributeTemplate,
from numba.core.typing import collections
@infer_global(list)
class ListBuiltin(AbstractTemplate):

    def generic(self, args, kws):
        assert not kws
        if args:
            iterable, = args
            if isinstance(iterable, types.IterableType):
                dtype = iterable.iterator_type.yield_type
                return signature(types.List(dtype), iterable)
        else:
            return signature(types.List(types.undefined))