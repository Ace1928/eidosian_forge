import numpy as np
from warnings import warn
from ._optimize import OptimizeResult, OptimizeWarning, _check_unknown_options
from ._linprog_util import _postsolve
def _linprog_simplex(c, c0, A, b, callback, postsolve_args, maxiter=1000, tol=1e-09, disp=False, bland=False, **unknown_options):
    """
    Minimize a linear objective function subject to linear equality and
    non-negativity constraints using the two phase simplex method.
    Linear programming is intended to solve problems of the following form:

    Minimize::

        c @ x

    Subject to::

        A @ x == b
            x >= 0

    User-facing documentation is in _linprog_doc.py.

    Parameters
    ----------
    c : 1-D array
        Coefficients of the linear objective function to be minimized.
    c0 : float
        Constant term in objective function due to fixed (and eliminated)
        variables. (Purely for display.)
    A : 2-D array
        2-D array such that ``A @ x``, gives the values of the equality
        constraints at ``x``.
    b : 1-D array
        1-D array of values representing the right hand side of each equality
        constraint (row) in ``A``.
    callback : callable, optional
        If a callback function is provided, it will be called within each
        iteration of the algorithm. The callback function must accept a single
        `scipy.optimize.OptimizeResult` consisting of the following fields:

            x : 1-D array
                Current solution vector
            fun : float
                Current value of the objective function
            success : bool
                True when an algorithm has completed successfully.
            slack : 1-D array
                The values of the slack variables. Each slack variable
                corresponds to an inequality constraint. If the slack is zero,
                the corresponding constraint is active.
            con : 1-D array
                The (nominally zero) residuals of the equality constraints,
                that is, ``b - A_eq @ x``
            phase : int
                The phase of the algorithm being executed.
            status : int
                An integer representing the status of the optimization::

                     0 : Algorithm proceeding nominally
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

    Options
    -------
    maxiter : int
       The maximum number of iterations to perform.
    disp : bool
        If True, print exit status message to sys.stdout
    tol : float
        The tolerance which determines when a solution is "close enough" to
        zero in Phase 1 to be considered a basic feasible solution or close
        enough to positive to serve as an optimal solution.
    bland : bool
        If True, use Bland's anti-cycling rule [3]_ to choose pivots to
        prevent cycling. If False, choose pivots which should lead to a
        converged solution more quickly. The latter method is subject to
        cycling (non-convergence) in rare instances.
    unknown_options : dict
        Optional arguments not used by this particular solver. If
        `unknown_options` is non-empty a warning is issued listing all
        unused options.

    Returns
    -------
    x : 1-D array
        Solution vector.
    status : int
        An integer representing the exit status of the optimization::

         0 : Optimization terminated successfully
         1 : Iteration limit reached
         2 : Problem appears to be infeasible
         3 : Problem appears to be unbounded
         4 : Serious numerical difficulties encountered

    message : str
        A string descriptor of the exit status of the optimization.
    iteration : int
        The number of iterations taken to solve the problem.

    References
    ----------
    .. [1] Dantzig, George B., Linear programming and extensions. Rand
           Corporation Research Study Princeton Univ. Press, Princeton, NJ,
           1963
    .. [2] Hillier, S.H. and Lieberman, G.J. (1995), "Introduction to
           Mathematical Programming", McGraw-Hill, Chapter 4.
    .. [3] Bland, Robert G. New finite pivoting rules for the simplex method.
           Mathematics of Operations Research (2), 1977: pp. 103-107.


    Notes
    -----
    The expected problem formulation differs between the top level ``linprog``
    module and the method specific solvers. The method specific solvers expect a
    problem in standard form:

    Minimize::

        c @ x

    Subject to::

        A @ x == b
            x >= 0

    Whereas the top level ``linprog`` module expects a problem of form:

    Minimize::

        c @ x

    Subject to::

        A_ub @ x <= b_ub
        A_eq @ x == b_eq
         lb <= x <= ub

    where ``lb = 0`` and ``ub = None`` unless set in ``bounds``.

    The original problem contains equality, upper-bound and variable constraints
    whereas the method specific solver requires equality constraints and
    variable non-negativity.

    ``linprog`` module converts the original problem to standard form by
    converting the simple bounds to upper bound constraints, introducing
    non-negative slack variables for inequality constraints, and expressing
    unbounded variables as the difference between two non-negative variables.
    """
    _check_unknown_options(unknown_options)
    status = 0
    messages = {0: 'Optimization terminated successfully.', 1: 'Iteration limit reached.', 2: 'Optimization failed. Unable to find a feasible starting point.', 3: 'Optimization failed. The problem appears to be unbounded.', 4: 'Optimization failed. Singular matrix encountered.'}
    n, m = A.shape
    is_negative_constraint = np.less(b, 0)
    A[is_negative_constraint] *= -1
    b[is_negative_constraint] *= -1
    av = np.arange(n) + m
    basis = av.copy()
    row_constraints = np.hstack((A, np.eye(n), b[:, np.newaxis]))
    row_objective = np.hstack((c, np.zeros(n), c0))
    row_pseudo_objective = -row_constraints.sum(axis=0)
    row_pseudo_objective[av] = 0
    T = np.vstack((row_constraints, row_objective, row_pseudo_objective))
    nit1, status = _solve_simplex(T, n, basis, callback=callback, postsolve_args=postsolve_args, maxiter=maxiter, tol=tol, phase=1, bland=bland)
    nit2 = nit1
    if abs(T[-1, -1]) < tol:
        T = T[:-1, :]
        T = np.delete(T, av, 1)
    else:
        status = 2
        messages[status] = "Phase 1 of the simplex method failed to find a feasible solution. The pseudo-objective function evaluates to {0:.1e} which exceeds the required tolerance of {1} for a solution to be considered 'close enough' to zero to be a basic solution. Consider increasing the tolerance to be greater than {0:.1e}. If this tolerance is unacceptably  large the problem may be infeasible.".format(abs(T[-1, -1]), tol)
    if status == 0:
        nit2, status = _solve_simplex(T, n, basis, callback=callback, postsolve_args=postsolve_args, maxiter=maxiter, tol=tol, phase=2, bland=bland, nit0=nit1)
    solution = np.zeros(n + m)
    solution[basis[:n]] = T[:n, -1]
    x = solution[:m]
    return (x, status, messages[status], int(nit2))