import itertools
import numpy as np
import operator
from numba.core import types, errors
from numba import prange
from numba.parfors.parfor import internal_prange
from numba.core.typing.templates import (AttributeTemplate, ConcreteTemplate,
from numba.cpython.builtins import get_type_min_value, get_type_max_value
from numba.core.extending import (
@infer_global(operator.add)
class TupleAdd(AbstractTemplate):

    def generic(self, args, kws):
        if len(args) == 2:
            a, b = args
            if isinstance(a, types.BaseTuple) and isinstance(b, types.BaseTuple) and (not isinstance(a, types.BaseNamedTuple)) and (not isinstance(b, types.BaseNamedTuple)):
                res = types.BaseTuple.from_types(tuple(a) + tuple(b))
                return signature(res, a, b)