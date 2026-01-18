import warnings
from collections import namedtuple
import operator
from . import _zeros
from ._optimize import OptimizeResult, _call_callback_maybe_halt
import numpy as np
def _compute_divided_differences(xvals, fvals, N=None, full=True, forward=True):
    """Return a matrix of divided differences for the xvals, fvals pairs

    DD[i, j] = f[x_{i-j}, ..., x_i] for 0 <= j <= i

    If full is False, just return the main diagonal(or last row):
      f[a], f[a, b] and f[a, b, c].
    If forward is False, return f[c], f[b, c], f[a, b, c]."""
    if full:
        if forward:
            xvals = np.asarray(xvals)
        else:
            xvals = np.array(xvals)[::-1]
        M = len(xvals)
        N = M if N is None else min(N, M)
        DD = np.zeros([M, N])
        DD[:, 0] = fvals[:]
        for i in range(1, N):
            DD[i:, i] = np.diff(DD[i - 1:, i - 1]) / (xvals[i:] - xvals[:M - i])
        return DD
    xvals = np.asarray(xvals)
    dd = np.array(fvals)
    row = np.array(fvals)
    idx2Use = 0 if forward else -1
    dd[0] = fvals[idx2Use]
    for i in range(1, len(xvals)):
        denom = xvals[i:i + len(row) - 1] - xvals[:len(row) - 1]
        row = np.diff(row)[:] / denom
        dd[i] = row[idx2Use]
    return dd