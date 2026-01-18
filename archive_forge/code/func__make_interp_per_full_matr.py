import operator
from math import prod
import numpy as np
from scipy._lib._util import normalize_axis_index
from scipy.linalg import (get_lapack_funcs, LinAlgError,
from scipy.optimize import minimize_scalar
from . import _bspl
from . import _fitpack_impl
from scipy.sparse import csr_array
from scipy.special import poch
from itertools import combinations
def _make_interp_per_full_matr(x, y, t, k):
    """
    Returns a solution of a system for B-spline interpolation with periodic
    boundary conditions. First ``k - 1`` rows of matrix are conditions of
    periodicity (continuity of ``k - 1`` derivatives at the boundary points).
    Last ``n`` rows are interpolation conditions.
    RHS is ``k - 1`` zeros and ``n`` ordinates in this case.

    Parameters
    ----------
    x : 1-D array, shape (n,)
        Values of x - coordinate of a given set of points.
    y : 1-D array, shape (n,)
        Values of y - coordinate of a given set of points.
    t : 1-D array, shape(n+2*k,)
        Vector of knots.
    k : int
        The maximum degree of spline

    Returns
    -------
    c : 1-D array, shape (n+k-1,)
        B-spline coefficients

    Notes
    -----
    ``t`` is supposed to be taken on circle.

    """
    x, y, t = map(np.asarray, (x, y, t))
    n = x.size
    matr = np.zeros((n + k - 1, n + k - 1))
    for i in range(k - 1):
        bb = _bspl.evaluate_all_bspl(t, k, x[0], k, nu=i + 1)
        matr[i, :k + 1] += bb
        bb = _bspl.evaluate_all_bspl(t, k, x[-1], n + k - 1, nu=i + 1)[:-1]
        matr[i, -k:] -= bb
    for i in range(n):
        xval = x[i]
        if xval == t[k]:
            left = k
        else:
            left = np.searchsorted(t, xval) - 1
        bb = _bspl.evaluate_all_bspl(t, k, xval, left)
        matr[i + k - 1, left - k:left + 1] = bb
    b = np.r_[[0] * (k - 1), y]
    c = solve(matr, b)
    return c