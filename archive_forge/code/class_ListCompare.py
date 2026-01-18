import operator
from numba.core import types
from .templates import (ConcreteTemplate, AbstractTemplate, AttributeTemplate,
from numba.core.typing import collections
class ListCompare(AbstractTemplate):

    def generic(self, args, kws):
        [lhs, rhs] = args
        if isinstance(lhs, types.List) and isinstance(rhs, types.List):
            res = self.context.resolve_function_type(self.key, (lhs.dtype, rhs.dtype), {})
            if res is not None:
                return signature(types.boolean, lhs, rhs)