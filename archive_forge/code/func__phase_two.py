import numpy as np
from numpy.linalg import LinAlgError
from scipy.linalg import solve
from ._optimize import _check_unknown_options
from ._bglu_dense import LU
from ._bglu_dense import BGLU as BGLU
from ._linprog_util import _postsolve
from ._optimize import OptimizeResult
def _phase_two(c, A, x, b, callback, postsolve_args, maxiter, tol, disp, maxupdate, mast, pivot, iteration=0, phase_one_n=None):
    """
    The heart of the simplex method. Beginning with a basic feasible solution,
    moves to adjacent basic feasible solutions successively lower reduced cost.
    Terminates when there are no basic feasible solutions with lower reduced
    cost or if the problem is determined to be unbounded.

    This implementation follows the revised simplex method based on LU
    decomposition. Rather than maintaining a tableau or an inverse of the
    basis matrix, we keep a factorization of the basis matrix that allows
    efficient solution of linear systems while avoiding stability issues
    associated with inverted matrices.
    """
    m, n = A.shape
    status = 0
    a = np.arange(n)
    ab = np.arange(m)
    if maxupdate:
        B = BGLU(A, b, maxupdate, mast)
    else:
        B = LU(A, b)
    for iteration in range(iteration, maxiter):
        if disp or callback is not None:
            _display_and_callback(phase_one_n, x, postsolve_args, status, iteration, disp, callback)
        bl = np.zeros(len(a), dtype=bool)
        bl[b] = 1
        xb = x[b]
        cb = c[b]
        try:
            v = B.solve(cb, transposed=True)
        except LinAlgError:
            status = 4
            break
        c_hat = c - v.dot(A)
        c_hat = c_hat[~bl]
        if np.all(c_hat >= -tol):
            break
        j = _select_enter_pivot(c_hat, bl, a, rule=pivot, tol=tol)
        u = B.solve(A[:, j])
        i = u > tol
        if not np.any(i):
            status = 3
            break
        th = xb[i] / u[i]
        l = np.argmin(th)
        th_star = th[l]
        x[b] = x[b] - th_star * u
        x[j] = th_star
        B.update(ab[i][l], j)
        b = B.b
    else:
        iteration += 1
        status = 1
        if disp or callback is not None:
            _display_and_callback(phase_one_n, x, postsolve_args, status, iteration, disp, callback)
    return (x, b, status, iteration)