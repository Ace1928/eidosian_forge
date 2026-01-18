from itertools import product
import operator
from numba.core import types
from numba.core.typing.templates import (AttributeTemplate, ConcreteTemplate,
from numba.np import npdatetime_helpers
from numba.np.numpy_support import numpy_version
@infer_global(operator.sub)
class DatetimeMinusDatetime(AbstractTemplate):
    key = operator.sub

    def generic(self, args, kws):
        if len(args) == 1:
            return
        left, right = args
        if isinstance(left, types.NPDatetime) and isinstance(right, types.NPDatetime):
            unit = npdatetime_helpers.get_best_unit(left.unit, right.unit)
            return signature(types.NPTimedelta(unit), left, right)