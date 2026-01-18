import itertools
import numpy as np
import operator
from numba.core import types, errors
from numba import prange
from numba.parfors.parfor import internal_prange
from numba.core.typing.templates import (AttributeTemplate, ConcreteTemplate,
from numba.cpython.builtins import get_type_min_value, get_type_max_value
from numba.core.extending import (
@infer_global(next)
class Next(AbstractTemplate):

    def generic(self, args, kws):
        assert not kws
        if len(args) == 1:
            it = args[0]
            if isinstance(it, types.IteratorType):
                return signature(it.yield_type, *args)