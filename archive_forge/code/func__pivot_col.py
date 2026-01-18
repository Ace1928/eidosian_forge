import numpy as np
from warnings import warn
from ._optimize import OptimizeResult, OptimizeWarning, _check_unknown_options
from ._linprog_util import _postsolve
def _pivot_col(T, tol=1e-09, bland=False):
    """
    Given a linear programming simplex tableau, determine the column
    of the variable to enter the basis.

    Parameters
    ----------
    T : 2-D array
        A 2-D array representing the simplex tableau, T, corresponding to the
        linear programming problem. It should have the form:

        [[A[0, 0], A[0, 1], ..., A[0, n_total], b[0]],
         [A[1, 0], A[1, 1], ..., A[1, n_total], b[1]],
         .
         .
         .
         [A[m, 0], A[m, 1], ..., A[m, n_total], b[m]],
         [c[0],   c[1], ...,   c[n_total],    0]]

        for a Phase 2 problem, or the form:

        [[A[0, 0], A[0, 1], ..., A[0, n_total], b[0]],
         [A[1, 0], A[1, 1], ..., A[1, n_total], b[1]],
         .
         .
         .
         [A[m, 0], A[m, 1], ..., A[m, n_total], b[m]],
         [c[0],   c[1], ...,   c[n_total],   0],
         [c'[0],  c'[1], ...,  c'[n_total],  0]]

         for a Phase 1 problem (a problem in which a basic feasible solution is
         sought prior to maximizing the actual objective. ``T`` is modified in
         place by ``_solve_simplex``.
    tol : float
        Elements in the objective row larger than -tol will not be considered
        for pivoting. Nominally this value is zero, but numerical issues
        cause a tolerance about zero to be necessary.
    bland : bool
        If True, use Bland's rule for selection of the column (select the
        first column with a negative coefficient in the objective row,
        regardless of magnitude).

    Returns
    -------
    status: bool
        True if a suitable pivot column was found, otherwise False.
        A return of False indicates that the linear programming simplex
        algorithm is complete.
    col: int
        The index of the column of the pivot element.
        If status is False, col will be returned as nan.
    """
    ma = np.ma.masked_where(T[-1, :-1] >= -tol, T[-1, :-1], copy=False)
    if ma.count() == 0:
        return (False, np.nan)
    if bland:
        return (True, np.nonzero(np.logical_not(np.atleast_1d(ma.mask)))[0][0])
    return (True, np.ma.nonzero(ma == ma.min())[0][0])