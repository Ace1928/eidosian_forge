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
class StaticGetItemTuple(AbstractTemplate):
    key = 'static_getitem'

    def generic(self, args, kws):
        tup, idx = args
        ret = None
        if not isinstance(tup, types.BaseTuple):
            return
        if isinstance(idx, int):
            try:
                ret = tup.types[idx]
            except IndexError:
                raise errors.NumbaIndexError('tuple index out of range')
        elif isinstance(idx, slice):
            ret = types.BaseTuple.from_types(tup.types[idx])
        if ret is not None:
            sig = signature(ret, *args)
            return sig