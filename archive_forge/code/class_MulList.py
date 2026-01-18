import operator
from numba.core import types
from .templates import (ConcreteTemplate, AbstractTemplate, AttributeTemplate,
from numba.core.typing import collections
@infer_global(operator.mul)
class MulList(AbstractTemplate):

    def generic(self, args, kws):
        a, b = args
        if isinstance(a, types.List) and isinstance(b, types.Integer):
            return signature(a, a, types.intp)
        elif isinstance(a, types.Integer) and isinstance(b, types.List):
            return signature(b, types.intp, b)