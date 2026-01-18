from itertools import product
import operator
from numba.core import types
from numba.core.typing.templates import (AttributeTemplate, ConcreteTemplate,
from numba.np import npdatetime_helpers
from numba.np.numpy_support import numpy_version
class TimedeltaDivOp(AbstractTemplate):

    def generic(self, args, kws):
        """
        (timedelta64, {int, float}) -> timedelta64
        (timedelta64, timedelta64) -> float
        """
        left, right = args
        if not isinstance(left, types.NPTimedelta):
            return
        if isinstance(right, types.NPTimedelta):
            if npdatetime_helpers.can_cast_timedelta_units(left.unit, right.unit) or npdatetime_helpers.can_cast_timedelta_units(right.unit, left.unit):
                return signature(types.float64, left, right)
        elif isinstance(right, types.Float):
            return signature(left, left, right)
        elif isinstance(right, types.Integer):
            return signature(left, left, types.int64)