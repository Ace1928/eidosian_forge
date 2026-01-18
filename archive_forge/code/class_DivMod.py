import itertools
import numpy as np
import operator
from numba.core import types, errors
from numba import prange
from numba.parfors.parfor import internal_prange
from numba.core.typing.templates import (AttributeTemplate, ConcreteTemplate,
from numba.cpython.builtins import get_type_min_value, get_type_max_value
from numba.core.extending import (
@infer_global(divmod)
class DivMod(ConcreteTemplate):
    _tys = machine_ints + sorted(types.real_domain)
    cases = [signature(types.UniTuple(ty, 2), ty, ty) for ty in _tys]