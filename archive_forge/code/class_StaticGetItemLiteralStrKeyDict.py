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
class StaticGetItemLiteralStrKeyDict(AbstractTemplate):
    key = 'static_getitem'

    def generic(self, args, kws):
        tup, idx = args
        ret = None
        if not isinstance(tup, types.LiteralStrKeyDict):
            return
        if isinstance(idx, str):
            if idx in tup.fields:
                lookup = tup.fields.index(idx)
            else:
                raise errors.NumbaKeyError(f"Key '{idx}' is not in dict.")
            ret = tup.types[lookup]
        if ret is not None:
            sig = signature(ret, *args)
            return sig