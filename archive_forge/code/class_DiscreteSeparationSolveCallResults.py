class DiscreteSeparationSolveCallResults:
    """
    Container for results of solve attempt for single separation
    problem.

    Parameters
    ----------
    solved_globally : bool
        True if separation problems solved to global optimality,
        False otherwise.
    solver_call_results : dict
        Mapping from discrete uncertainty set scenario list
        indexes to solver call results for separation problems
        subject to the scenarios.
    performance_constraint : Constraint
        Separation problem performance constraint for which
        `self` was generated.

    Attributes
    ----------
    solved_globally
    scenario_indexes
    solver_call_results
    performance_constraint
    time_out
    subsolver_error
    """

    def __init__(self, solved_globally, solver_call_results=None, performance_constraint=None):
        """Initialize self (see class docstring)."""
        self.solved_globally = solved_globally
        self.solver_call_results = solver_call_results
        self.performance_constraint = performance_constraint

    @property
    def time_out(self):
        """
        bool : True if there is a time out status for at least one of
        the ``SeparationSolveCallResults`` objects listed in `self`,
        False otherwise.
        """
        return any((res.time_out for res in self.solver_call_results.values()))

    @property
    def subsolver_error(self):
        """
        bool : True if there is a subsolver error status for at least
        one of the the ``SeparationSolveCallResults`` objects listed
        in `self`, False otherwise.
        """
        return any((res.subsolver_error for res in self.solver_call_results.values()))

    def evaluate_total_solve_time(self, evaluator_func, **evaluator_func_kwargs):
        """
        Evaluate total time required by subordinate solvers
        for separation problem of interest.

        Parameters
        ----------
        evaluator_func : callable
            Solve time evaluator function.
            This callable should accept an object of type
            ``pyomo.opt.results.SolveResults``, and
            return a float equal to the time required.
        **evaluator_func_kwargs : dict, optional
            Keyword arguments to evaluator function.

        Returns
        -------
        float
            Total time spent by solvers.
        """
        return sum((solver_call_res.evaluate_total_solve_time(evaluator_func) for solver_call_res in self.solver_call_results.values()))