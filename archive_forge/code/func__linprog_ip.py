import numpy as np
import scipy as sp
import scipy.sparse as sps
from warnings import warn
from scipy.linalg import LinAlgError
from ._optimize import OptimizeWarning, OptimizeResult, _check_unknown_options
from ._linprog_util import _postsolve
def _linprog_ip(c, c0, A, b, callback, postsolve_args, maxiter=1000, tol=1e-08, disp=False, alpha0=0.99995, beta=0.1, sparse=False, lstsq=False, sym_pos=True, cholesky=None, pc=True, ip=False, permc_spec='MMD_AT_PLUS_A', **unknown_options):
    """
    Minimize a linear objective function subject to linear
    equality and non-negativity constraints using the interior point method
    of [4]_. Linear programming is intended to solve problems
    of the following form:

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
        Callback function to be executed once per iteration.
    postsolve_args : tuple
        Data needed by _postsolve to convert the solution to the standard-form
        problem into the solution to the original problem.

    Options
    -------
    maxiter : int (default = 1000)
        The maximum number of iterations of the algorithm.
    tol : float (default = 1e-8)
        Termination tolerance to be used for all termination criteria;
        see [4]_ Section 4.5.
    disp : bool (default = False)
        Set to ``True`` if indicators of optimization status are to be printed
        to the console each iteration.
    alpha0 : float (default = 0.99995)
        The maximal step size for Mehrota's predictor-corrector search
        direction; see :math:`\\beta_{3}` of [4]_ Table 8.1.
    beta : float (default = 0.1)
        The desired reduction of the path parameter :math:`\\mu` (see [6]_)
        when Mehrota's predictor-corrector is not in use (uncommon).
    sparse : bool (default = False)
        Set to ``True`` if the problem is to be treated as sparse after
        presolve. If either ``A_eq`` or ``A_ub`` is a sparse matrix,
        this option will automatically be set ``True``, and the problem
        will be treated as sparse even during presolve. If your constraint
        matrices contain mostly zeros and the problem is not very small (less
        than about 100 constraints or variables), consider setting ``True``
        or providing ``A_eq`` and ``A_ub`` as sparse matrices.
    lstsq : bool (default = False)
        Set to ``True`` if the problem is expected to be very poorly
        conditioned. This should always be left ``False`` unless severe
        numerical difficulties are encountered. Leave this at the default
        unless you receive a warning message suggesting otherwise.
    sym_pos : bool (default = True)
        Leave ``True`` if the problem is expected to yield a well conditioned
        symmetric positive definite normal equation matrix
        (almost always). Leave this at the default unless you receive
        a warning message suggesting otherwise.
    cholesky : bool (default = True)
        Set to ``True`` if the normal equations are to be solved by explicit
        Cholesky decomposition followed by explicit forward/backward
        substitution. This is typically faster for problems
        that are numerically well-behaved.
    pc : bool (default = True)
        Leave ``True`` if the predictor-corrector method of Mehrota is to be
        used. This is almost always (if not always) beneficial.
    ip : bool (default = False)
        Set to ``True`` if the improved initial point suggestion due to [4]_
        Section 4.3 is desired. Whether this is beneficial or not
        depends on the problem.
    permc_spec : str (default = 'MMD_AT_PLUS_A')
        (Has effect only with ``sparse = True``, ``lstsq = False``, ``sym_pos =
        True``, and no SuiteSparse.)
        A matrix is factorized in each iteration of the algorithm.
        This option specifies how to permute the columns of the matrix for
        sparsity preservation. Acceptable values are:

        - ``NATURAL``: natural ordering.
        - ``MMD_ATA``: minimum degree ordering on the structure of A^T A.
        - ``MMD_AT_PLUS_A``: minimum degree ordering on the structure of A^T+A.
        - ``COLAMD``: approximate minimum degree column ordering.

        This option can impact the convergence of the
        interior point algorithm; test different values to determine which
        performs best for your problem. For more information, refer to
        ``scipy.sparse.linalg.splu``.
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

    Notes
    -----
    This method implements the algorithm outlined in [4]_ with ideas from [8]_
    and a structure inspired by the simpler methods of [6]_.

    The primal-dual path following method begins with initial 'guesses' of
    the primal and dual variables of the standard form problem and iteratively
    attempts to solve the (nonlinear) Karush-Kuhn-Tucker conditions for the
    problem with a gradually reduced logarithmic barrier term added to the
    objective. This particular implementation uses a homogeneous self-dual
    formulation, which provides certificates of infeasibility or unboundedness
    where applicable.

    The default initial point for the primal and dual variables is that
    defined in [4]_ Section 4.4 Equation 8.22. Optionally (by setting initial
    point option ``ip=True``), an alternate (potentially improved) starting
    point can be calculated according to the additional recommendations of
    [4]_ Section 4.4.

    A search direction is calculated using the predictor-corrector method
    (single correction) proposed by Mehrota and detailed in [4]_ Section 4.1.
    (A potential improvement would be to implement the method of multiple
    corrections described in [4]_ Section 4.2.) In practice, this is
    accomplished by solving the normal equations, [4]_ Section 5.1 Equations
    8.31 and 8.32, derived from the Newton equations [4]_ Section 5 Equations
    8.25 (compare to [4]_ Section 4 Equations 8.6-8.8). The advantage of
    solving the normal equations rather than 8.25 directly is that the
    matrices involved are symmetric positive definite, so Cholesky
    decomposition can be used rather than the more expensive LU factorization.

    With default options, the solver used to perform the factorization depends
    on third-party software availability and the conditioning of the problem.

    For dense problems, solvers are tried in the following order:

    1. ``scipy.linalg.cho_factor``

    2. ``scipy.linalg.solve`` with option ``sym_pos=True``

    3. ``scipy.linalg.solve`` with option ``sym_pos=False``

    4. ``scipy.linalg.lstsq``

    For sparse problems:

    1. ``sksparse.cholmod.cholesky`` (if scikit-sparse and SuiteSparse are installed)

    2. ``scipy.sparse.linalg.factorized``
        (if scikit-umfpack and SuiteSparse are installed)

    3. ``scipy.sparse.linalg.splu`` (which uses SuperLU distributed with SciPy)

    4. ``scipy.sparse.linalg.lsqr``

    If the solver fails for any reason, successively more robust (but slower)
    solvers are attempted in the order indicated. Attempting, failing, and
    re-starting factorization can be time consuming, so if the problem is
    numerically challenging, options can be set to  bypass solvers that are
    failing. Setting ``cholesky=False`` skips to solver 2,
    ``sym_pos=False`` skips to solver 3, and ``lstsq=True`` skips
    to solver 4 for both sparse and dense problems.

    Potential improvements for combatting issues associated with dense
    columns in otherwise sparse problems are outlined in [4]_ Section 5.3 and
    [10]_ Section 4.1-4.2; the latter also discusses the alleviation of
    accuracy issues associated with the substitution approach to free
    variables.

    After calculating the search direction, the maximum possible step size
    that does not activate the non-negativity constraints is calculated, and
    the smaller of this step size and unity is applied (as in [4]_ Section
    4.1.) [4]_ Section 4.3 suggests improvements for choosing the step size.

    The new point is tested according to the termination conditions of [4]_
    Section 4.5. The same tolerance, which can be set using the ``tol`` option,
    is used for all checks. (A potential improvement would be to expose
    the different tolerances to be set independently.) If optimality,
    unboundedness, or infeasibility is detected, the solve procedure
    terminates; otherwise it repeats.

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


    References
    ----------
    .. [4] Andersen, Erling D., and Knud D. Andersen. "The MOSEK interior point
           optimizer for linear programming: an implementation of the
           homogeneous algorithm." High performance optimization. Springer US,
           2000. 197-232.
    .. [6] Freund, Robert M. "Primal-Dual Interior-Point Methods for Linear
           Programming based on Newton's Method." Unpublished Course Notes,
           March 2004. Available 2/25/2017 at
           https://ocw.mit.edu/courses/sloan-school-of-management/15-084j-nonlinear-programming-spring-2004/lecture-notes/lec14_int_pt_mthd.pdf
    .. [8] Andersen, Erling D., and Knud D. Andersen. "Presolving in linear
           programming." Mathematical Programming 71.2 (1995): 221-245.
    .. [9] Bertsimas, Dimitris, and J. Tsitsiklis. "Introduction to linear
           programming." Athena Scientific 1 (1997): 997.
    .. [10] Andersen, Erling D., et al. Implementation of interior point methods
            for large scale linear programming. HEC/Universite de Geneve, 1996.

    """
    _check_unknown_options(unknown_options)
    if (cholesky or cholesky is None) and sparse and (not has_cholmod):
        if cholesky:
            warn('Sparse cholesky is only available with scikit-sparse. Setting `cholesky = False`', OptimizeWarning, stacklevel=3)
        cholesky = False
    if sparse and lstsq:
        warn("Option combination 'sparse':True and 'lstsq':True is not recommended.", OptimizeWarning, stacklevel=3)
    if lstsq and cholesky:
        warn("Invalid option combination 'lstsq':True and 'cholesky':True; option 'cholesky' has no effect when 'lstsq' is set True.", OptimizeWarning, stacklevel=3)
    valid_permc_spec = ('NATURAL', 'MMD_ATA', 'MMD_AT_PLUS_A', 'COLAMD')
    if permc_spec.upper() not in valid_permc_spec:
        warn("Invalid permc_spec option: '" + str(permc_spec) + "'. Acceptable values are 'NATURAL', 'MMD_ATA', 'MMD_AT_PLUS_A', and 'COLAMD'. Reverting to default.", OptimizeWarning, stacklevel=3)
        permc_spec = 'MMD_AT_PLUS_A'
    if not sym_pos and cholesky:
        raise ValueError("Invalid option combination 'sym_pos':False and 'cholesky':True: Cholesky decomposition is only possible for symmetric positive definite matrices.")
    cholesky = cholesky or (cholesky is None and sym_pos and (not lstsq))
    x, status, message, iteration = _ip_hsd(A, b, c, c0, alpha0, beta, maxiter, disp, tol, sparse, lstsq, sym_pos, cholesky, pc, ip, permc_spec, callback, postsolve_args)
    return (x, status, message, iteration)