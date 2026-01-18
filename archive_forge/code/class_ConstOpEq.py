import itertools
import numpy as np
import operator
from numba.core import types, errors
from numba import prange
from numba.parfors.parfor import internal_prange
from numba.core.typing.templates import (AttributeTemplate, ConcreteTemplate,
from numba.cpython.builtins import get_type_min_value, get_type_max_value
from numba.core.extending import (
@infer_global(operator.eq)
class ConstOpEq(AbstractTemplate):

    def generic(self, args, kws):
        assert not kws
        arg1, arg2 = args
        if isinstance(arg1, types.Literal) and isinstance(arg2, types.Literal):
            return signature(types.boolean, arg1, arg2)