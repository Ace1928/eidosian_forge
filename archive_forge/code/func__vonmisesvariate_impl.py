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
def _vonmisesvariate_impl(_random):

    def _impl(mu, kappa):
        """Circular data distribution.  Taken from CPython.
        Note the algorithm in Python 2.6 and Numpy is different:
        http://bugs.python.org/issue17141
        """
        if kappa <= 1e-06:
            return 2.0 * math.pi * _random()
        s = 0.5 / kappa
        r = s + math.sqrt(1.0 + s * s)
        while 1:
            u1 = _random()
            z = math.cos(math.pi * u1)
            d = z / (r + z)
            u2 = _random()
            if u2 < 1.0 - d * d or u2 <= (1.0 - d) * math.exp(d):
                break
        q = 1.0 / r
        f = (q + z) / (1.0 + q * z)
        u3 = _random()
        if u3 > 0.5:
            theta = (mu + math.acos(f)) % (2.0 * math.pi)
        else:
            theta = (mu - math.acos(f)) % (2.0 * math.pi)
        return theta
    return _impl