from itertools import product
import operator
from numba.core import types
from numba.core.typing.templates import (AttributeTemplate, ConcreteTemplate,
from numba.np import npdatetime_helpers
from numba.np.numpy_support import numpy_version
class TimedeltaUnaryOp(AbstractTemplate):

    def generic(self, args, kws):
        if len(args) == 2:
            return
        op, = args
        if not isinstance(op, types.NPTimedelta):
            return
        return signature(op, op)