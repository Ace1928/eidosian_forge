import warnings
from collections import namedtuple
import operator
from . import _zeros
from ._optimize import OptimizeResult, _call_callback_maybe_halt
import numpy as np
def _differentiate_weights(work, n):
    fac = work.fac.astype(np.float64)
    if fac != _differentiate_weights.fac:
        _differentiate_weights.central = []
        _differentiate_weights.right = []
        _differentiate_weights.fac = fac
    if len(_differentiate_weights.central) != 2 * n + 1:
        i = np.arange(-n, n + 1)
        p = np.abs(i) - 1.0
        s = np.sign(i)
        h = s / fac ** p
        A = np.vander(h, increasing=True).T
        b = np.zeros(2 * n + 1)
        b[1] = 1
        weights = np.linalg.solve(A, b)
        weights[n] = 0
        for i in range(n):
            weights[-i - 1] = -weights[i]
        _differentiate_weights.central = weights
        i = np.arange(2 * n + 1)
        p = i - 1.0
        s = np.sign(i)
        h = s / np.sqrt(fac) ** p
        A = np.vander(h, increasing=True).T
        b = np.zeros(2 * n + 1)
        b[1] = 1
        weights = np.linalg.solve(A, b)
        _differentiate_weights.right = weights
    return (_differentiate_weights.central.astype(work.dtype, copy=False), _differentiate_weights.right.astype(work.dtype, copy=False))