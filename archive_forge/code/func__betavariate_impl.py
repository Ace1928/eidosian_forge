import math
import random
import numpy as np
from llvmlite import ir
from numba.core.cgutils import is_nonelike
from numba.core.extending import intrinsic, overload, register_jitable
from numba.core.imputils import (Registry, impl_ret_untracked,
from numba.core.typing import signature
from numba.core import types, cgutils
from numba.np import arrayobj
from numba.core.errors import NumbaTypeError
def _betavariate_impl(gamma):

    def _impl(alpha, beta):
        """Beta distribution.  Taken from CPython.
        """
        y = gamma(alpha, 1.0)
        if y == 0.0:
            return 0.0
        else:
            return y / (y + gamma(beta, 1.0))
    return _impl