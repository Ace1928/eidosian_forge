import numpy as np
from numpy.linalg import LinAlgError
from scipy.linalg import solve
from ._optimize import _check_unknown_options
from ._bglu_dense import LU
from ._bglu_dense import BGLU as BGLU
from ._linprog_util import _postsolve
from ._optimize import OptimizeResult

    Solve the following linear programming problem via a two-phase
    revised simplex algorithm.::

        minimize:     c @ x

        subject to:  A @ x == b
                     0 <= x < oo

    User-facing documentation is in _linprog_doc.py.

    Parameters
    ----------
    c : 1-D array
        Coefficients of the linear objective function to be minimized.
    c0 : float
        Constant term in objective function due to fixed (and eliminated)
        variables. (Currently unused.)
    A : 2-D array
        2-D array which, when matrix-multiplied by ``x``, gives the values of
        the equality constraints at ``x``.
    b : 1-D array
        1-D array of values representing the RHS of each equality constraint
        (row) in ``A_eq``.
    x0 : 1-D array, optional
        Starting values of the independent variables, which will be refined by
        the optimization algorithm. For the revised simplex method, these must
        correspond with a basic feasible solution.
    callback : callable, optional
        If a callback function is provided, it will be called within each
        iteration of the algorithm. The callback function must accept a single
        `scipy.optimize.OptimizeResult` consisting of the following fields:

            x : 1-D array
                Current solution vector.
            fun : float
                Current value of the objective function ``c @ x``.
            success : bool
                True only when an algorithm has completed successfully,
                so this is always False as the callback function is called
                only while the algorithm is still iterating.
            slack : 1-D array
                The values of the slack variables. Each slack variable
                corresponds to an inequality constraint. If the slack is zero,
                the corresponding constraint is active.
            con : 1-D array
                The (nominally zero) residuals of the equality constraints,
                that is, ``b - A_eq @ x``.
            phase : int
                The phase of the algorithm being executed.
            status : int
                For revised simplex, this is always 0 because if a different
                status is detected, the algorithm terminates.
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
       The maximum number of iterations to perform in either phase.
    tol : float
        The tolerance which determines when a solution is "close enough" to
        zero in Phase 1 to be considered a basic feasible solution or close
        enough to positive to serve as an optimal solution.
    disp : bool
        Set to ``True`` if indicators of optimization status are to be printed
        to the console each iteration.
    maxupdate : int
        The maximum number of updates performed on the LU factorization.
        After this many updates is reached, the basis matrix is factorized
        from scratch.
    mast : bool
        Minimize Amortized Solve Time. If enabled, the average time to solve
        a linear system using the basis factorization is measured. Typically,
        the average solve time will decrease with each successive solve after
        initial factorization, as factorization takes much more time than the
        solve operation (and updates). Eventually, however, the updated
        factorization becomes sufficiently complex that the average solve time
        begins to increase. When this is detected, the basis is refactorized
        from scratch. Enable this option to maximize speed at the risk of
        nondeterministic behavior. Ignored if ``maxupdate`` is 0.
    pivot : "mrc" or "bland"
        Pivot rule: Minimum Reduced Cost (default) or Bland's rule. Choose
        Bland's rule if iteration limit is reached and cycling is suspected.
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
         4 : Numerical difficulties encountered
         5 : No constraints; turn presolve on
         6 : Guess x0 cannot be converted to a basic feasible solution

    message : str
        A string descriptor of the exit status of the optimization.
    iteration : int
        The number of iterations taken to solve the problem.
    