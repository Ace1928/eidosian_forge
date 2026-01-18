def _linprog_rs_doc(c, A_ub=None, b_ub=None, A_eq=None, b_eq=None, bounds=None, method='interior-point', callback=None, x0=None, maxiter=5000, disp=False, presolve=True, tol=1e-12, autoscale=False, rr=True, maxupdate=10, mast=False, pivot='mrc', **unknown_options):
    """
    Linear programming: minimize a linear objective function subject to linear
    equality and inequality constraints using the revised simplex method.

    .. deprecated:: 1.9.0
        `method='revised simplex'` will be removed in SciPy 1.11.0.
        It is replaced by `method='highs'` because the latter is
        faster and more robust.

    Linear programming solves problems of the following form:

    .. math::

        \\min_x \\ & c^T x \\\\
        \\mbox{such that} \\ & A_{ub} x \\leq b_{ub},\\\\
        & A_{eq} x = b_{eq},\\\\
        & l \\leq x \\leq u ,

    where :math:`x` is a vector of decision variables; :math:`c`,
    :math:`b_{ub}`, :math:`b_{eq}`, :math:`l`, and :math:`u` are vectors; and
    :math:`A_{ub}` and :math:`A_{eq}` are matrices.

    Alternatively, that's:

    minimize::

        c @ x

    such that::

        A_ub @ x <= b_ub
        A_eq @ x == b_eq
        lb <= x <= ub

    Note that by default ``lb = 0`` and ``ub = None`` unless specified with
    ``bounds``.

    Parameters
    ----------
    c : 1-D array
        The coefficients of the linear objective function to be minimized.
    A_ub : 2-D array, optional
        The inequality constraint matrix. Each row of ``A_ub`` specifies the
        coefficients of a linear inequality constraint on ``x``.
    b_ub : 1-D array, optional
        The inequality constraint vector. Each element represents an
        upper bound on the corresponding value of ``A_ub @ x``.
    A_eq : 2-D array, optional
        The equality constraint matrix. Each row of ``A_eq`` specifies the
        coefficients of a linear equality constraint on ``x``.
    b_eq : 1-D array, optional
        The equality constraint vector. Each element of ``A_eq @ x`` must equal
        the corresponding element of ``b_eq``.
    bounds : sequence, optional
        A sequence of ``(min, max)`` pairs for each element in ``x``, defining
        the minimum and maximum values of that decision variable. Use ``None``
        to indicate that there is no bound. By default, bounds are
        ``(0, None)`` (all decision variables are non-negative).
        If a single tuple ``(min, max)`` is provided, then ``min`` and
        ``max`` will serve as bounds for all decision variables.
    method : str
        This is the method-specific documentation for 'revised simplex'.
        :ref:`'highs' <optimize.linprog-highs>`,
        :ref:`'highs-ds' <optimize.linprog-highs-ds>`,
        :ref:`'highs-ipm' <optimize.linprog-highs-ipm>`,
        :ref:`'interior-point' <optimize.linprog-interior-point>` (default),
        and :ref:`'simplex' <optimize.linprog-simplex>` (legacy)
        are also available.
    callback : callable, optional
        Callback function to be executed once per iteration.
    x0 : 1-D array, optional
        Guess values of the decision variables, which will be refined by
        the optimization algorithm. This argument is currently used only by the
        'revised simplex' method, and can only be used if `x0` represents a
        basic feasible solution.

    Options
    -------
    maxiter : int (default: 5000)
       The maximum number of iterations to perform in either phase.
    disp : bool (default: False)
        Set to ``True`` if indicators of optimization status are to be printed
        to the console each iteration.
    presolve : bool (default: True)
        Presolve attempts to identify trivial infeasibilities,
        identify trivial unboundedness, and simplify the problem before
        sending it to the main solver. It is generally recommended
        to keep the default setting ``True``; set to ``False`` if
        presolve is to be disabled.
    tol : float (default: 1e-12)
        The tolerance which determines when a solution is "close enough" to
        zero in Phase 1 to be considered a basic feasible solution or close
        enough to positive to serve as an optimal solution.
    autoscale : bool (default: False)
        Set to ``True`` to automatically perform equilibration.
        Consider using this option if the numerical values in the
        constraints are separated by several orders of magnitude.
    rr : bool (default: True)
        Set to ``False`` to disable automatic redundancy removal.
    maxupdate : int (default: 10)
        The maximum number of updates performed on the LU factorization.
        After this many updates is reached, the basis matrix is factorized
        from scratch.
    mast : bool (default: False)
        Minimize Amortized Solve Time. If enabled, the average time to solve
        a linear system using the basis factorization is measured. Typically,
        the average solve time will decrease with each successive solve after
        initial factorization, as factorization takes much more time than the
        solve operation (and updates). Eventually, however, the updated
        factorization becomes sufficiently complex that the average solve time
        begins to increase. When this is detected, the basis is refactorized
        from scratch. Enable this option to maximize speed at the risk of
        nondeterministic behavior. Ignored if ``maxupdate`` is 0.
    pivot : "mrc" or "bland" (default: "mrc")
        Pivot rule: Minimum Reduced Cost ("mrc") or Bland's rule ("bland").
        Choose Bland's rule if iteration limit is reached and cycling is
        suspected.
    unknown_options : dict
        Optional arguments not used by this particular solver. If
        `unknown_options` is non-empty a warning is issued listing all
        unused options.

    Returns
    -------
    res : OptimizeResult
        A :class:`scipy.optimize.OptimizeResult` consisting of the fields:

        x : 1-D array
            The values of the decision variables that minimizes the
            objective function while satisfying the constraints.
        fun : float
            The optimal value of the objective function ``c @ x``.
        slack : 1-D array
            The (nominally positive) values of the slack variables,
            ``b_ub - A_ub @ x``.
        con : 1-D array
            The (nominally zero) residuals of the equality constraints,
            ``b_eq - A_eq @ x``.
        success : bool
            ``True`` when the algorithm succeeds in finding an optimal
            solution.
        status : int
            An integer representing the exit status of the algorithm.

            ``0`` : Optimization terminated successfully.

            ``1`` : Iteration limit reached.

            ``2`` : Problem appears to be infeasible.

            ``3`` : Problem appears to be unbounded.

            ``4`` : Numerical difficulties encountered.

            ``5`` : Problem has no constraints; turn presolve on.

            ``6`` : Invalid guess provided.

        message : str
            A string descriptor of the exit status of the algorithm.
        nit : int
            The total number of iterations performed in all phases.


    Notes
    -----
    Method *revised simplex* uses the revised simplex method as described in
    [9]_, except that a factorization [11]_ of the basis matrix, rather than
    its inverse, is efficiently maintained and used to solve the linear systems
    at each iteration of the algorithm.

    References
    ----------
    .. [9] Bertsimas, Dimitris, and J. Tsitsiklis. "Introduction to linear
           programming." Athena Scientific 1 (1997): 997.
    .. [11] Bartels, Richard H. "A stabilization of the simplex method."
            Journal in  Numerische Mathematik 16.5 (1971): 414-434.
    """
    pass