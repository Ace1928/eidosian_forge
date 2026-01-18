from itertools import product
import operator
from numba.core import types
from numba.core.typing.templates import (AttributeTemplate, ConcreteTemplate,
from numba.np import npdatetime_helpers
from numba.np.numpy_support import numpy_version
@infer_global(operator.add)
@infer_global(operator.iadd)
class DatetimePlusTimedelta(AbstractTemplate):
    key = operator.add

    def generic(self, args, kws):
        if len(args) == 1:
            return
        left, right = args
        if isinstance(right, types.NPTimedelta):
            dt = left
            td = right
        elif isinstance(left, types.NPTimedelta):
            dt = right
            td = left
        else:
            return
        if isinstance(dt, types.NPDatetime):
            unit = npdatetime_helpers.combine_datetime_timedelta_units(dt.unit, td.unit)
            if unit is not None:
                return signature(types.NPDatetime(unit), left, right)