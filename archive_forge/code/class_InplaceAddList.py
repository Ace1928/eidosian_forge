import operator
from numba.core import types
from .templates import (ConcreteTemplate, AbstractTemplate, AttributeTemplate,
from numba.core.typing import collections
@infer_global(operator.iadd)
class InplaceAddList(AbstractTemplate):

    def generic(self, args, kws):
        if len(args) == 2:
            a, b = args
            if isinstance(a, types.List) and isinstance(b, types.List):
                if self.context.can_convert(b.dtype, a.dtype):
                    return signature(a, a, b)