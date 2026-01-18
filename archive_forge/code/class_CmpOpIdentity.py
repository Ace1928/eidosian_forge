import itertools
import numpy as np
import operator
from numba.core import types, errors
from numba import prange
from numba.parfors.parfor import internal_prange
from numba.core.typing.templates import (AttributeTemplate, ConcreteTemplate,
from numba.cpython.builtins import get_type_min_value, get_type_max_value
from numba.core.extending import (
class CmpOpIdentity(AbstractTemplate):

    def generic(self, args, kws):
        [lhs, rhs] = args
        return signature(types.boolean, lhs, rhs)