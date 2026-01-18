import numpy as np
from warnings import warn
from ._optimize import OptimizeResult, OptimizeWarning, _check_unknown_options
from ._linprog_util import _postsolve
def _pivot_row(T, basis, pivcol, phase, tol=1e-09, bland=False):
    """
    Given a linear programming simplex tableau, determine the row for the
    pivot operation.

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

         for a Phase 1 problem (a Problem in which a basic feasible solution is
         sought prior to maximizing the actual objective. ``T`` is modified in
         place by ``_solve_simplex``.
    basis : array
        A list of the current basic variables.
    pivcol : int
        The index of the pivot column.
    phase : int
        The phase of the simplex algorithm (1 or 2).
    tol : float
        Elements in the pivot column smaller than tol will not be considered
        for pivoting. Nominally this value is zero, but numerical issues
        cause a tolerance about zero to be necessary.
    bland : bool
        If True, use Bland's rule for selection of the row (if more than one
        row can be used, choose the one with the lowest variable index).

    Returns
    -------
    status: bool
        True if a suitable pivot row was found, otherwise False. A return
        of False indicates that the linear programming problem is unbounded.
    row: int
        The index of the row of the pivot element. If status is False, row
        will be returned as nan.
    """
    if phase == 1:
        k = 2
    else:
        k = 1
    ma = np.ma.masked_where(T[:-k, pivcol] <= tol, T[:-k, pivcol], copy=False)
    if ma.count() == 0:
        return (False, np.nan)
    mb = np.ma.masked_where(T[:-k, pivcol] <= tol, T[:-k, -1], copy=False)
    q = mb / ma
    min_rows = np.ma.nonzero(q == q.min())[0]
    if bland:
        return (True, min_rows[np.argmin(np.take(basis, min_rows))])
    return (True, min_rows[0])