import numpy as _np
import functools
from scipy.linalg import _fblas
from scipy.linalg._fblas import *  # noqa: E402, F403
def _memoize_get_funcs(func):
    """
    Memoized fast path for _get_funcs instances
    """
    memo = {}
    func.memo = memo

    @functools.wraps(func)
    def getter(names, arrays=(), dtype=None, ilp64=False):
        key = (names, dtype, ilp64)
        for array in arrays:
            key += (array.dtype.char, array.flags.fortran)
        try:
            value = memo.get(key)
        except TypeError:
            key = None
            value = None
        if value is not None:
            return value
        value = func(names, arrays, dtype, ilp64)
        if key is not None:
            memo[key] = value
        return value
    return getter