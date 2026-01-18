from itertools import product
import operator
from numba.core import types
from numba.core.typing.templates import (AttributeTemplate, ConcreteTemplate,
from numba.np import npdatetime_helpers
from numba.np.numpy_support import numpy_version
class TimedeltaCmpOp(AbstractTemplate):

    def generic(self, args, kws):
        left, right = args
        if not all((isinstance(tp, types.NPTimedelta) for tp in args)):
            return
        return signature(types.boolean, left, right)