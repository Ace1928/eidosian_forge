import operator
from numba.core import types
from .templates import (ConcreteTemplate, AbstractTemplate, AttributeTemplate,
from numba.core.typing import collections
class SetComparison(AbstractTemplate):

    def generic(self, args, kws):
        if len(args) != 2:
            return
        a, b = args
        if isinstance(a, types.Set) and isinstance(b, types.Set) and (a == b):
            return signature(types.boolean, *args)