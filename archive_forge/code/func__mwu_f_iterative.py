import numpy as np
from collections import namedtuple
from scipy import special
from scipy import stats
from ._axis_nan_policy import _axis_nan_policy_factory
def _mwu_f_iterative(m, n, k, fmnks):
    """Iterative implementation of function of [3] Theorem 2.5"""

    def _base_case(m, n, k):
        """Base cases from recursive version"""
        if fmnks.get((m, n, k), -1) >= 0:
            return fmnks[m, n, k]
        elif k < 0 or m < 0 or n < 0 or (k > m * n):
            return 0
        elif k == 0 and m >= 0 and (n >= 0):
            return 1
        return None
    stack = [(m, n, k)]
    fmnk = None
    while stack:
        m, n, k = stack.pop()
        fmnk = _base_case(m, n, k)
        if fmnk is not None:
            fmnks[m, n, k] = fmnk
            continue
        f1 = _base_case(m - 1, n, k - n)
        f2 = _base_case(m, n - 1, k)
        if f1 is not None and f2 is not None:
            fmnk = f1 + f2
            fmnks[m, n, k] = fmnk
            continue
        stack.append((m, n, k))
        if f1 is None:
            stack.append((m - 1, n, k - n))
        if f2 is None:
            stack.append((m, n - 1, k))
    return fmnks