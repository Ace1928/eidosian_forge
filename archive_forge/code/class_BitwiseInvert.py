import itertools
import numpy as np
import operator
from numba.core import types, errors
from numba import prange
from numba.parfors.parfor import internal_prange
from numba.core.typing.templates import (AttributeTemplate, ConcreteTemplate,
from numba.cpython.builtins import get_type_min_value, get_type_max_value
from numba.core.extending import (
@infer_global(operator.invert)
class BitwiseInvert(ConcreteTemplate):
    cases = [signature(types.boolean, types.boolean)]
    cases += [signature(choose_result_int(op), op) for op in sorted(types.unsigned_domain)]
    cases += [signature(choose_result_int(op), op) for op in sorted(types.signed_domain)]
    unsafe_casting = False