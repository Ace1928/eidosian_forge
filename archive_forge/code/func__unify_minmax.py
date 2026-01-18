import itertools
import numpy as np
import operator
from numba.core import types, errors
from numba import prange
from numba.parfors.parfor import internal_prange
from numba.core.typing.templates import (AttributeTemplate, ConcreteTemplate,
from numba.cpython.builtins import get_type_min_value, get_type_max_value
from numba.core.extending import (
def _unify_minmax(self, tys):
    for ty in tys:
        if not isinstance(ty, (types.Number, types.NPDatetime, types.NPTimedelta)):
            return
    return self.context.unify_types(*tys)