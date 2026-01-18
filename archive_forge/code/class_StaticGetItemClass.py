import itertools
import numpy as np
import operator
from numba.core import types, errors
from numba import prange
from numba.parfors.parfor import internal_prange
from numba.core.typing.templates import (AttributeTemplate, ConcreteTemplate,
from numba.cpython.builtins import get_type_min_value, get_type_max_value
from numba.core.extending import (
@infer
class StaticGetItemClass(AbstractTemplate):
    """This handles the "static_getitem" when a Numba type is subscripted e.g:
    var = typed.List.empty_list(float64[::1, :])
    It only allows this on simple numerical types. Compound types, like
    records, are not supported.
    """
    key = 'static_getitem'

    def generic(self, args, kws):
        clazz, idx = args
        if not isinstance(clazz, types.NumberClass):
            return
        ret = clazz.dtype[idx]
        sig = signature(ret, *args)
        return sig