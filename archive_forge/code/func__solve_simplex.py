import numpy as np
from warnings import warn
from ._optimize import OptimizeResult, OptimizeWarning, _check_unknown_options
from ._linprog_util import _postsolve
def _solve_simplex(T, n, basis, callback, postsolve_args, maxiter=1000, tol=1e-09, phase=2, bland=False, nit0=0):
    """
    Solve a linear programming problem in "standard form" using the Simplex
    Method. Linear Programming is intended to solve the following problem form:

    Minimize::

        c @ x

    Subject to::

        A @ x == b
            x >= 0

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
    n : int
        The number of true variables in the problem.
    basis : 1-D array
        An array of the indices of the basic variables, such that basis[i]
        contains the column corresponding to the basic variable for row i.
        Basis is modified in place by _solve_simplex
    callback : callable, optional
        If a callback function is provided, it will be called within each
        iteration of the algorithm. The callback must accept a
        `scipy.optimize.OptimizeResult` consisting of the following fields:

            x : 1-D array
                Current solution vector
            fun : float
                Current value of the objective function
            success : bool
                True only when a phase has completed successfully. This
                will be False for most iterations.
            slack : 1-D array
                The values of the slack variables. Each slack variable
                corresponds to an inequality constraint. If the slack is zero,
                the corresponding constraint is active.
            con : 1-D array
                The (nominally zero) residuals of the equality constraints,
                that is, ``b - A_eq @ x``
            phase : int
                The phase of the optimization being executed. In phase 1 a basic
                feasible solution is sought and the T has an additional row
                representing an alternate objective function.
            status : int
                An integer representing the exit status of the optimization::

                     0 : Optimization terminated successfully
                     1 : Iteration limit reached
                     2 : Problem appears to be infeasible
                     3 : Problem appears to be unbounded
                     4 : Serious numerical difficulties encountered

            nit : int
                The number of iterations performed.
            message : str
                A string descriptor of the exit status of the optimization.
    postsolve_args : tuple
        Data needed by _postsolve to convert the solution to the standard-form
        problem into the solution to the original problem.
    maxiter : int
        The maximum number of iterations to perform before aborting the
        optimization.
    tol : float
        The tolerance which determines when a solution is "close enough" to
        zero in Phase 1 to be considered a basic feasible solution or close
        enough to positive to serve as an optimal solution.
    phase : int
        The phase of the optimization being executed. In phase 1 a basic
        feasible solution is sought and the T has an additional row
        representing an alternate objective function.
    bland : bool
        If True, choose pivots using Bland's rule [3]_. In problems which
        fail to converge due to cycling, using Bland's rule can provide
        convergence at the expense of a less optimal path about the simplex.
    nit0 : int
        The initial iteration number used to keep an accurate iteration total
        in a two-phase problem.

    Returns
    -------
    nit : int
        The number of iterations. Used to keep an accurate iteration total
        in the two-phase problem.
    status : int
        An integer representing the exit status of the optimization::

         0 : Optimization terminated successfully
         1 : Iteration limit reached
         2 : Problem appears to be infeasible
         3 : Problem appears to be unbounded
         4 : Serious numerical difficulties encountered

    """
    nit = nit0
    status = 0
    message = ''
    complete = False
    if phase == 1:
        m = T.shape[1] - 2
    elif phase == 2:
        m = T.shape[1] - 1
    else:
        raise ValueError("Argument 'phase' to _solve_simplex must be 1 or 2")
    if phase == 2:
        for pivrow in [row for row in range(basis.size) if basis[row] > T.shape[1] - 2]:
            non_zero_row = [col for col in range(T.shape[1] - 1) if abs(T[pivrow, col]) > tol]
            if len(non_zero_row) > 0:
                pivcol = non_zero_row[0]
                _apply_pivot(T, basis, pivrow, pivcol, tol)
                nit += 1
    if len(basis[:m]) == 0:
        solution = np.empty(T.shape[1] - 1, dtype=np.float64)
    else:
        solution = np.empty(max(T.shape[1] - 1, max(basis[:m]) + 1), dtype=np.float64)
    while not complete:
        pivcol_found, pivcol = _pivot_col(T, tol, bland)
        if not pivcol_found:
            pivcol = np.nan
            pivrow = np.nan
            status = 0
            complete = True
        else:
            pivrow_found, pivrow = _pivot_row(T, basis, pivcol, phase, tol, bland)
            if not pivrow_found:
                status = 3
                complete = True
        if callback is not None:
            solution[:] = 0
            solution[basis[:n]] = T[:n, -1]
            x = solution[:m]
            x, fun, slack, con = _postsolve(x, postsolve_args)
            res = OptimizeResult({'x': x, 'fun': fun, 'slack': slack, 'con': con, 'status': status, 'message': message, 'nit': nit, 'success': status == 0 and complete, 'phase': phase, 'complete': complete})
            callback(res)
        if not complete:
            if nit >= maxiter:
                status = 1
                complete = True
            else:
                _apply_pivot(T, basis, pivrow, pivcol, tol)
                nit += 1
    return (nit, status)