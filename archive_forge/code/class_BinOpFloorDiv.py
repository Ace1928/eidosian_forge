import itertools
import numpy as np
import operator
from numba.core import types, errors
from numba import prange
from numba.parfors.parfor import internal_prange
from numba.core.typing.templates import (AttributeTemplate, ConcreteTemplate,
from numba.cpython.builtins import get_type_min_value, get_type_max_value
from numba.core.extending import (
@infer_global(operator.ifloordiv)
class BinOpFloorDiv(ConcreteTemplate):
    cases = list(integer_binop_cases)
    cases += [signature(op, op, op) for op in sorted(types.real_domain)]