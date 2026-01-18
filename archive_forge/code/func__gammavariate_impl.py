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
def _gammavariate_impl(_random):

    def _impl(alpha, beta):
        """Gamma distribution.  Taken from CPython.
        """
        SG_MAGICCONST = 1.0 + math.log(4.5)
        if alpha <= 0.0 or beta <= 0.0:
            raise ValueError('gammavariate: alpha and beta must be > 0.0')
        if alpha > 1.0:
            ainv = math.sqrt(2.0 * alpha - 1.0)
            bbb = alpha - math.log(4.0)
            ccc = alpha + ainv
            while 1:
                u1 = _random()
                if not 1e-07 < u1 < 0.9999999:
                    continue
                u2 = 1.0 - _random()
                v = math.log(u1 / (1.0 - u1)) / ainv
                x = alpha * math.exp(v)
                z = u1 * u1 * u2
                r = bbb + ccc * v - x
                if r + SG_MAGICCONST - 4.5 * z >= 0.0 or r >= math.log(z):
                    return x * beta
        elif alpha == 1.0:
            return -math.log(1.0 - _random()) * beta
        else:
            while 1:
                u = _random()
                b = (math.e + alpha) / math.e
                p = b * u
                if p <= 1.0:
                    x = p ** (1.0 / alpha)
                else:
                    x = -math.log((b - p) / alpha)
                u1 = _random()
                if p > 1.0:
                    if u1 <= x ** (alpha - 1.0):
                        break
                elif u1 <= math.exp(-x):
                    break
            return x * beta
    return _impl