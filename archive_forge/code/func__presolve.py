import numpy as np
import scipy.sparse as sps
from warnings import warn
from ._optimize import OptimizeWarning
from scipy.optimize._remove_redundancy import (
from collections import namedtuple
def _presolve(lp, rr, rr_method, tol=1e-09):
    """
    Given inputs for a linear programming problem in preferred format,
    presolve the problem: identify trivial infeasibilities, redundancies,
    and unboundedness, tighten bounds where possible, and eliminate fixed
    variables.

    Parameters
    ----------
    lp : A `scipy.optimize._linprog_util._LPProblem` consisting of the following fields:

        c : 1D array
            The coefficients of the linear objective function to be minimized.
        A_ub : 2D array, optional
            The inequality constraint matrix. Each row of ``A_ub`` specifies the
            coefficients of a linear inequality constraint on ``x``.
        b_ub : 1D array, optional
            The inequality constraint vector. Each element represents an
            upper bound on the corresponding value of ``A_ub @ x``.
        A_eq : 2D array, optional
            The equality constraint matrix. Each row of ``A_eq`` specifies the
            coefficients of a linear equality constraint on ``x``.
        b_eq : 1D array, optional
            The equality constraint vector. Each element of ``A_eq @ x`` must equal
            the corresponding element of ``b_eq``.
        bounds : 2D array
            The bounds of ``x``, as ``min`` and ``max`` pairs, one for each of the N
            elements of ``x``. The N x 2 array contains lower bounds in the first
            column and upper bounds in the 2nd. Unbounded variables have lower
            bound -np.inf and/or upper bound np.inf.
        x0 : 1D array, optional
            Guess values of the decision variables, which will be refined by
            the optimization algorithm. This argument is currently used only by the
            'revised simplex' method, and can only be used if `x0` represents a
            basic feasible solution.

    rr : bool
        If ``True`` attempts to eliminate any redundant rows in ``A_eq``.
        Set False if ``A_eq`` is known to be of full row rank, or if you are
        looking for a potential speedup (at the expense of reliability).
    rr_method : string
        Method used to identify and remove redundant rows from the
        equality constraint matrix after presolve.
    tol : float
        The tolerance which determines when a solution is "close enough" to
        zero in Phase 1 to be considered a basic feasible solution or close
        enough to positive to serve as an optimal solution.

    Returns
    -------
    lp : A `scipy.optimize._linprog_util._LPProblem` consisting of the following fields:

        c : 1D array
            The coefficients of the linear objective function to be minimized.
        A_ub : 2D array, optional
            The inequality constraint matrix. Each row of ``A_ub`` specifies the
            coefficients of a linear inequality constraint on ``x``.
        b_ub : 1D array, optional
            The inequality constraint vector. Each element represents an
            upper bound on the corresponding value of ``A_ub @ x``.
        A_eq : 2D array, optional
            The equality constraint matrix. Each row of ``A_eq`` specifies the
            coefficients of a linear equality constraint on ``x``.
        b_eq : 1D array, optional
            The equality constraint vector. Each element of ``A_eq @ x`` must equal
            the corresponding element of ``b_eq``.
        bounds : 2D array
            The bounds of ``x``, as ``min`` and ``max`` pairs, possibly tightened.
        x0 : 1D array, optional
            Guess values of the decision variables, which will be refined by
            the optimization algorithm. This argument is currently used only by the
            'revised simplex' method, and can only be used if `x0` represents a
            basic feasible solution.

    c0 : 1D array
        Constant term in objective function due to fixed (and eliminated)
        variables.
    x : 1D array
        Solution vector (when the solution is trivial and can be determined
        in presolve)
    revstack: list of functions
        the functions in the list reverse the operations of _presolve()
        the function signature is x_org = f(x_mod), where x_mod is the result
        of a presolve step and x_org the value at the start of the step
        (currently, the revstack contains only one function)
    complete: bool
        Whether the solution is complete (solved or determined to be infeasible
        or unbounded in presolve)
    status : int
        An integer representing the exit status of the optimization::

         0 : Optimization terminated successfully
         1 : Iteration limit reached
         2 : Problem appears to be infeasible
         3 : Problem appears to be unbounded
         4 : Serious numerical difficulties encountered

    message : str
        A string descriptor of the exit status of the optimization.

    References
    ----------
    .. [5] Andersen, Erling D. "Finding all linearly dependent rows in
           large-scale linear programming." Optimization Methods and Software
           6.3 (1995): 219-227.
    .. [8] Andersen, Erling D., and Knud D. Andersen. "Presolving in linear
           programming." Mathematical Programming 71.2 (1995): 221-245.

    """
    c, A_ub, b_ub, A_eq, b_eq, bounds, x0, _ = lp
    revstack = []
    c0 = 0
    complete = False
    x = np.zeros(c.shape)
    status = 0
    message = ''
    lb = bounds[:, 0].copy()
    ub = bounds[:, 1].copy()
    m_eq, n = A_eq.shape
    m_ub, n = A_ub.shape
    if rr_method is not None and rr_method.lower() not in {'svd', 'pivot', 'id'}:
        message = "'" + str(rr_method) + "' is not a valid option for redundancy removal. Valid options are 'SVD', 'pivot', and 'ID'."
        raise ValueError(message)
    if sps.issparse(A_eq):
        A_eq = A_eq.tocsr()
        A_ub = A_ub.tocsr()

        def where(A):
            return A.nonzero()
        vstack = sps.vstack
    else:
        where = np.where
        vstack = np.vstack
    if np.any(ub < lb) or np.any(lb == np.inf) or np.any(ub == -np.inf):
        status = 2
        message = 'The problem is (trivially) infeasible since one or more upper bounds are smaller than the corresponding lower bounds, a lower bound is np.inf or an upper bound is -np.inf.'
        complete = True
        return (_LPProblem(c, A_ub, b_ub, A_eq, b_eq, bounds, x0), c0, x, revstack, complete, status, message)
    zero_row = np.array(np.sum(A_eq != 0, axis=1) == 0).flatten()
    if np.any(zero_row):
        if np.any(np.logical_and(zero_row, np.abs(b_eq) > tol)):
            status = 2
            message = 'The problem is (trivially) infeasible due to a row of zeros in the equality constraint matrix with a nonzero corresponding constraint value.'
            complete = True
            return (_LPProblem(c, A_ub, b_ub, A_eq, b_eq, bounds, x0), c0, x, revstack, complete, status, message)
        else:
            A_eq = A_eq[np.logical_not(zero_row), :]
            b_eq = b_eq[np.logical_not(zero_row)]
    zero_row = np.array(np.sum(A_ub != 0, axis=1) == 0).flatten()
    if np.any(zero_row):
        if np.any(np.logical_and(zero_row, b_ub < -tol)):
            status = 2
            message = 'The problem is (trivially) infeasible due to a row of zeros in the equality constraint matrix with a nonzero corresponding  constraint value.'
            complete = True
            return (_LPProblem(c, A_ub, b_ub, A_eq, b_eq, bounds, x0), c0, x, revstack, complete, status, message)
        else:
            A_ub = A_ub[np.logical_not(zero_row), :]
            b_ub = b_ub[np.logical_not(zero_row)]
    A = vstack((A_eq, A_ub))
    if A.shape[0] > 0:
        zero_col = np.array(np.sum(A != 0, axis=0) == 0).flatten()
        x[np.logical_and(zero_col, c < 0)] = ub[np.logical_and(zero_col, c < 0)]
        x[np.logical_and(zero_col, c > 0)] = lb[np.logical_and(zero_col, c > 0)]
        if np.any(np.isinf(x)):
            status = 3
            message = 'If feasible, the problem is (trivially) unbounded due  to a zero column in the constraint matrices. If you wish to check whether the problem is infeasible, turn presolve off.'
            complete = True
            return (_LPProblem(c, A_ub, b_ub, A_eq, b_eq, bounds, x0), c0, x, revstack, complete, status, message)
        lb[np.logical_and(zero_col, c < 0)] = ub[np.logical_and(zero_col, c < 0)]
        ub[np.logical_and(zero_col, c > 0)] = lb[np.logical_and(zero_col, c > 0)]
    singleton_row = np.array(np.sum(A_eq != 0, axis=1) == 1).flatten()
    rows = where(singleton_row)[0]
    cols = where(A_eq[rows, :])[1]
    if len(rows) > 0:
        for row, col in zip(rows, cols):
            val = b_eq[row] / A_eq[row, col]
            if not lb[col] - tol <= val <= ub[col] + tol:
                status = 2
                message = 'The problem is (trivially) infeasible because a singleton row in the equality constraints is inconsistent with the bounds.'
                complete = True
                return (_LPProblem(c, A_ub, b_ub, A_eq, b_eq, bounds, x0), c0, x, revstack, complete, status, message)
            else:
                lb[col] = val
                ub[col] = val
        A_eq = A_eq[np.logical_not(singleton_row), :]
        b_eq = b_eq[np.logical_not(singleton_row)]
    singleton_row = np.array(np.sum(A_ub != 0, axis=1) == 1).flatten()
    cols = where(A_ub[singleton_row, :])[1]
    rows = where(singleton_row)[0]
    if len(rows) > 0:
        for row, col in zip(rows, cols):
            val = b_ub[row] / A_ub[row, col]
            if A_ub[row, col] > 0:
                if val < lb[col] - tol:
                    complete = True
                elif val < ub[col]:
                    ub[col] = val
            elif val > ub[col] + tol:
                complete = True
            elif val > lb[col]:
                lb[col] = val
            if complete:
                status = 2
                message = 'The problem is (trivially) infeasible because a singleton row in the upper bound constraints is inconsistent with the bounds.'
                return (_LPProblem(c, A_ub, b_ub, A_eq, b_eq, bounds, x0), c0, x, revstack, complete, status, message)
        A_ub = A_ub[np.logical_not(singleton_row), :]
        b_ub = b_ub[np.logical_not(singleton_row)]
    i_f = np.abs(lb - ub) < tol
    i_nf = np.logical_not(i_f)
    if np.all(i_f):
        residual = b_eq - A_eq.dot(lb)
        slack = b_ub - A_ub.dot(lb)
        if A_ub.size > 0 and np.any(slack < 0) or (A_eq.size > 0 and (not np.allclose(residual, 0))):
            status = 2
            message = 'The problem is (trivially) infeasible because the bounds fix all variables to values inconsistent with the constraints'
            complete = True
            return (_LPProblem(c, A_ub, b_ub, A_eq, b_eq, bounds, x0), c0, x, revstack, complete, status, message)
    ub_mod = ub
    lb_mod = lb
    if np.any(i_f):
        c0 += c[i_f].dot(lb[i_f])
        b_eq = b_eq - A_eq[:, i_f].dot(lb[i_f])
        b_ub = b_ub - A_ub[:, i_f].dot(lb[i_f])
        c = c[i_nf]
        x_undo = lb[i_f]
        x = x[i_nf]
        if x0 is not None:
            x0 = x0[i_nf]
        A_eq = A_eq[:, i_nf]
        A_ub = A_ub[:, i_nf]
        lb_mod = lb[i_nf]
        ub_mod = ub[i_nf]

        def rev(x_mod):
            i = np.flatnonzero(i_f)
            N = len(i)
            index_offset = np.arange(N)
            insert_indices = i - index_offset
            x_rev = np.insert(x_mod.astype(float), insert_indices, x_undo)
            return x_rev
        revstack.append(rev)
    if A_eq.size == 0 and A_ub.size == 0:
        b_eq = np.array([])
        b_ub = np.array([])
        if c.size == 0:
            status = 0
            message = 'The solution was determined in presolve as there are no non-trivial constraints.'
        elif np.any(np.logical_and(c < 0, ub_mod == np.inf)) or np.any(np.logical_and(c > 0, lb_mod == -np.inf)):
            status = 3
            message = 'The problem is (trivially) unbounded because there are no non-trivial constraints and a) at least one decision variable is unbounded above and its corresponding cost is negative, or b) at least one decision variable is unbounded below and its corresponding cost is positive. '
        else:
            status = 0
            message = 'The solution was determined in presolve as there are no non-trivial constraints.'
        complete = True
        x[c < 0] = ub_mod[c < 0]
        x[c > 0] = lb_mod[c > 0]
        x_zero_c = ub_mod[c == 0]
        x_zero_c[np.isinf(x_zero_c)] = ub_mod[c == 0][np.isinf(x_zero_c)]
        x_zero_c[np.isinf(x_zero_c)] = 0
        x[c == 0] = x_zero_c
    bounds = np.hstack((lb_mod[:, np.newaxis], ub_mod[:, np.newaxis]))
    n_rows_A = A_eq.shape[0]
    redundancy_warning = 'A_eq does not appear to be of full row rank. To improve performance, check the problem formulation for redundant equality constraints.'
    if sps.issparse(A_eq):
        if rr and A_eq.size > 0:
            rr_res = _remove_redundancy_pivot_sparse(A_eq, b_eq)
            A_eq, b_eq, status, message = rr_res
            if A_eq.shape[0] < n_rows_A:
                warn(redundancy_warning, OptimizeWarning, stacklevel=1)
            if status != 0:
                complete = True
        return (_LPProblem(c, A_ub, b_ub, A_eq, b_eq, bounds, x0), c0, x, revstack, complete, status, message)
    small_nullspace = 5
    if rr and A_eq.size > 0:
        try:
            rank = np.linalg.matrix_rank(A_eq)
        except Exception:
            rank = 0
    if rr and A_eq.size > 0 and (rank < A_eq.shape[0]):
        warn(redundancy_warning, OptimizeWarning, stacklevel=3)
        dim_row_nullspace = A_eq.shape[0] - rank
        if rr_method is None:
            if dim_row_nullspace <= small_nullspace:
                rr_res = _remove_redundancy_svd(A_eq, b_eq)
                A_eq, b_eq, status, message = rr_res
            if dim_row_nullspace > small_nullspace or status == 4:
                rr_res = _remove_redundancy_pivot_dense(A_eq, b_eq)
                A_eq, b_eq, status, message = rr_res
        else:
            rr_method = rr_method.lower()
            if rr_method == 'svd':
                rr_res = _remove_redundancy_svd(A_eq, b_eq)
                A_eq, b_eq, status, message = rr_res
            elif rr_method == 'pivot':
                rr_res = _remove_redundancy_pivot_dense(A_eq, b_eq)
                A_eq, b_eq, status, message = rr_res
            elif rr_method == 'id':
                rr_res = _remove_redundancy_id(A_eq, b_eq, rank)
                A_eq, b_eq, status, message = rr_res
            else:
                pass
        if A_eq.shape[0] < rank:
            message = 'Due to numerical issues, redundant equality constraints could not be removed automatically. Try providing your constraint matrices as sparse matrices to activate sparse presolve, try turning off redundancy removal, or try turning off presolve altogether.'
            status = 4
        if status != 0:
            complete = True
    return (_LPProblem(c, A_ub, b_ub, A_eq, b_eq, bounds, x0), c0, x, revstack, complete, status, message)