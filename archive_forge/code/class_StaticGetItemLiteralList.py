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
class StaticGetItemLiteralList(AbstractTemplate):
    key = 'static_getitem'

    def generic(self, args, kws):
        tup, idx = args
        ret = None
        if not isinstance(tup, types.LiteralList):
            return
        if isinstance(idx, int):
            ret = tup.types[idx]
        if ret is not None:
            sig = signature(ret, *args)
            return sig