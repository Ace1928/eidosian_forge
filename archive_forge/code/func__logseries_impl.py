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
def _logseries_impl(p):
    """Numpy's algorithm for logseries()."""
    if p <= 0.0 or p > 1.0:
        raise ValueError('logseries(): p outside of (0, 1]')
    r = math.log(1.0 - p)
    while 1:
        V = np.random.random()
        if V >= p:
            return 1
        U = np.random.random()
        q = 1.0 - math.exp(r * U)
        if V <= q * q:
            return np.int64(1.0 + math.log(V) / math.log(q))
        elif V >= q:
            return 1
        else:
            return 2