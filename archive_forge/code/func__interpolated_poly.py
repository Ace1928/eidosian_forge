import warnings
from collections import namedtuple
import operator
from . import _zeros
from ._optimize import OptimizeResult, _call_callback_maybe_halt
import numpy as np
def _interpolated_poly(xvals, fvals, x):
    """Compute p(x) for the polynomial passing through the specified locations.

    Use Neville's algorithm to compute p(x) where p is the minimal degree
    polynomial passing through the points xvals, fvals"""
    xvals = np.asarray(xvals)
    N = len(xvals)
    Q = np.zeros([N, N])
    D = np.zeros([N, N])
    Q[:, 0] = fvals[:]
    D[:, 0] = fvals[:]
    for k in range(1, N):
        alpha = D[k:, k - 1] - Q[k - 1:N - 1, k - 1]
        diffik = xvals[0:N - k] - xvals[k:N]
        Q[k:, k] = (xvals[k:] - x) / diffik * alpha
        D[k:, k] = (xvals[:N - k] - x) / diffik * alpha
    return np.sum(Q[-1, 1:]) + Q[-1, 0]