import inspect
import numpy as np
from ._optimize import OptimizeWarning, OptimizeResult
from warnings import warn
from ._highs._highs_wrapper import _highs_wrapper
from ._highs._highs_constants import (
from scipy.sparse import csc_matrix, vstack, issparse

    Solve the following linear programming problem using one of the HiGHS
    solvers:

    User-facing documentation is in _linprog_doc.py.

    Parameters
    ----------
    lp :  _LPProblem
        A ``scipy.optimize._linprog_util._LPProblem`` ``namedtuple``.
    solver : "ipm" or "simplex" or None
        Which HiGHS solver to use.  If ``None``, "simplex" will be used.

    Options
    -------
    maxiter : int
        The maximum number of iterations to perform in either phase. For
        ``solver='ipm'``, this does not include the number of crossover
        iterations.  Default is the largest possible value for an ``int``
        on the platform.
    disp : bool
        Set to ``True`` if indicators of optimization status are to be printed
        to the console each iteration; default ``False``.
    time_limit : float
        The maximum time in seconds allotted to solve the problem; default is
        the largest possible value for a ``double`` on the platform.
    presolve : bool
        Presolve attempts to identify trivial infeasibilities,
        identify trivial unboundedness, and simplify the problem before
        sending it to the main solver. It is generally recommended
        to keep the default setting ``True``; set to ``False`` if presolve is
        to be disabled.
    dual_feasibility_tolerance : double
        Dual feasibility tolerance.  Default is 1e-07.
        The minimum of this and ``primal_feasibility_tolerance``
        is used for the feasibility tolerance when ``solver='ipm'``.
    primal_feasibility_tolerance : double
        Primal feasibility tolerance.  Default is 1e-07.
        The minimum of this and ``dual_feasibility_tolerance``
        is used for the feasibility tolerance when ``solver='ipm'``.
    ipm_optimality_tolerance : double
        Optimality tolerance for ``solver='ipm'``.  Default is 1e-08.
        Minimum possible value is 1e-12 and must be smaller than the largest
        possible value for a ``double`` on the platform.
    simplex_dual_edge_weight_strategy : str (default: None)
        Strategy for simplex dual edge weights. The default, ``None``,
        automatically selects one of the following.

        ``'dantzig'`` uses Dantzig's original strategy of choosing the most
        negative reduced cost.

        ``'devex'`` uses the strategy described in [15]_.

        ``steepest`` uses the exact steepest edge strategy as described in
        [16]_.

        ``'steepest-devex'`` begins with the exact steepest edge strategy
        until the computation is too costly or inexact and then switches to
        the devex method.

        Currently, using ``None`` always selects ``'steepest-devex'``, but this
        may change as new options become available.

    mip_max_nodes : int
        The maximum number of nodes allotted to solve the problem; default is
        the largest possible value for a ``HighsInt`` on the platform.
        Ignored if not using the MIP solver.
    unknown_options : dict
        Optional arguments not used by this particular solver. If
        ``unknown_options`` is non-empty, a warning is issued listing all
        unused options.

    Returns
    -------
    sol : dict
        A dictionary consisting of the fields:

            x : 1D array
                The values of the decision variables that minimizes the
                objective function while satisfying the constraints.
            fun : float
                The optimal value of the objective function ``c @ x``.
            slack : 1D array
                The (nominally positive) values of the slack,
                ``b_ub - A_ub @ x``.
            con : 1D array
                The (nominally zero) residuals of the equality constraints,
                ``b_eq - A_eq @ x``.
            success : bool
                ``True`` when the algorithm succeeds in finding an optimal
                solution.
            status : int
                An integer representing the exit status of the algorithm.

                ``0`` : Optimization terminated successfully.

                ``1`` : Iteration or time limit reached.

                ``2`` : Problem appears to be infeasible.

                ``3`` : Problem appears to be unbounded.

                ``4`` : The HiGHS solver ran into a problem.

            message : str
                A string descriptor of the exit status of the algorithm.
            nit : int
                The total number of iterations performed.
                For ``solver='simplex'``, this includes iterations in all
                phases. For ``solver='ipm'``, this does not include
                crossover iterations.
            crossover_nit : int
                The number of primal/dual pushes performed during the
                crossover routine for ``solver='ipm'``.  This is ``0``
                for ``solver='simplex'``.
            ineqlin : OptimizeResult
                Solution and sensitivity information corresponding to the
                inequality constraints, `b_ub`. A dictionary consisting of the
                fields:

                residual : np.ndnarray
                    The (nominally positive) values of the slack variables,
                    ``b_ub - A_ub @ x``.  This quantity is also commonly
                    referred to as "slack".

                marginals : np.ndarray
                    The sensitivity (partial derivative) of the objective
                    function with respect to the right-hand side of the
                    inequality constraints, `b_ub`.

            eqlin : OptimizeResult
                Solution and sensitivity information corresponding to the
                equality constraints, `b_eq`.  A dictionary consisting of the
                fields:

                residual : np.ndarray
                    The (nominally zero) residuals of the equality constraints,
                    ``b_eq - A_eq @ x``.

                marginals : np.ndarray
                    The sensitivity (partial derivative) of the objective
                    function with respect to the right-hand side of the
                    equality constraints, `b_eq`.

            lower, upper : OptimizeResult
                Solution and sensitivity information corresponding to the
                lower and upper bounds on decision variables, `bounds`.

                residual : np.ndarray
                    The (nominally positive) values of the quantity
                    ``x - lb`` (lower) or ``ub - x`` (upper).

                marginals : np.ndarray
                    The sensitivity (partial derivative) of the objective
                    function with respect to the lower and upper
                    `bounds`.

            mip_node_count : int
                The number of subproblems or "nodes" solved by the MILP
                solver. Only present when `integrality` is not `None`.

            mip_dual_bound : float
                The MILP solver's final estimate of the lower bound on the
                optimal solution. Only present when `integrality` is not
                `None`.

            mip_gap : float
                The difference between the final objective function value
                and the final dual bound, scaled by the final objective
                function value. Only present when `integrality` is not
                `None`.

    Notes
    -----
    The result fields `ineqlin`, `eqlin`, `lower`, and `upper` all contain
    `marginals`, or partial derivatives of the objective function with respect
    to the right-hand side of each constraint. These partial derivatives are
    also referred to as "Lagrange multipliers", "dual values", and
    "shadow prices". The sign convention of `marginals` is opposite that
    of Lagrange multipliers produced by many nonlinear solvers.

    References
    ----------
    .. [15] Harris, Paula MJ. "Pivot selection methods of the Devex LP code."
            Mathematical programming 5.1 (1973): 1-28.
    .. [16] Goldfarb, Donald, and John Ker Reid. "A practicable steepest-edge
            simplex algorithm." Mathematical Programming 12.1 (1977): 361-371.
    