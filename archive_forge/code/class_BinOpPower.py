import itertools
import numpy as np
import operator
from numba.core import types, errors
from numba import prange
from numba.parfors.parfor import internal_prange
from numba.core.typing.templates import (AttributeTemplate, ConcreteTemplate,
from numba.cpython.builtins import get_type_min_value, get_type_max_value
from numba.core.extending import (
@infer_global(operator.ipow)
class BinOpPower(ConcreteTemplate):
    cases = list(integer_binop_cases)
    cases += [signature(types.float32, types.float32, op) for op in (types.int32, types.int64, types.uint64)]
    cases += [signature(types.float64, types.float64, op) for op in (types.int32, types.int64, types.uint64)]
    cases += [signature(op, op, op) for op in sorted(types.real_domain)]
    cases += [signature(op, op, op) for op in sorted(types.complex_domain)]