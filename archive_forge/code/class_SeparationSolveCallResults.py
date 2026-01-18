class SeparationSolveCallResults:
    """
    Container for results of solve attempt for single separation
    problem.

    Parameters
    ----------
    solved_globally : bool
        True if separation problem was solved globally,
        False otherwise.
    results_list : list of pyomo.opt.results.SolverResults, optional
        Pyomo solver results for each subordinate optimizer invoked on
        the separation problem.
        For problems with non-discrete uncertainty set types,
        each entry corresponds to a single subordinate solver.
        For problems with discrete set types, the list may
        be empty (didn't need to use a subordinate solver to
        evaluate optimal separation solution), or the number
        of entries may be as high as the product of the number of
        subordinate local/global solvers provided (including backup)
        and the number of scenarios in the uncertainty set.
    scaled_violations : ComponentMap, optional
        Mapping from performance constraints to floats equal
        to their scaled violations by separation problem solution
        stored in this result.
    violating_param_realization : list of float, optional
        Uncertain parameter realization for reported separation
        problem solution.
    variable_values : ComponentMap, optional
        Second-stage DOF and state variable values for reported
        separation problem solution.
    found_violation : bool, optional
        True if violation of performance constraint (i.e. constraint
        expression value) by reported separation solution was found to
        exceed tolerance, False otherwise.
    time_out : bool, optional
        True if PyROS time limit reached attempting to solve the
        separation problem, False otherwise.
    subsolver_error : bool, optional
        True if subsolvers found to be unable to solve separation
        problem of interest, False otherwise.
    discrete_set_scenario_index : None or int, optional
        If discrete set used to solve the problem, index of
        `violating_param_realization` as listed in the
        `scenarios` attribute of a ``DiscreteScenarioSet``
        instance. If discrete set not used, pass None.

    Attributes
    ----------
    solved_globally
    results_list
    scaled_violations
    violating_param_realizations
    variable_values
    found_violation
    time_out
    subsolver_error
    discrete_set_scenario_index
    """

    def __init__(self, solved_globally, results_list=None, scaled_violations=None, violating_param_realization=None, variable_values=None, found_violation=None, time_out=None, subsolver_error=None, discrete_set_scenario_index=None):
        """Initialize self (see class docstring)."""
        self.results_list = results_list
        self.solved_globally = solved_globally
        self.scaled_violations = scaled_violations
        self.violating_param_realization = violating_param_realization
        self.variable_values = variable_values
        self.found_violation = found_violation
        self.time_out = time_out
        self.subsolver_error = subsolver_error
        self.discrete_set_scenario_index = discrete_set_scenario_index

    def termination_acceptable(self, acceptable_terminations):
        """
        Return True if termination condition for at least
        one result in `self.results_list` is in list
        of pre-specified acceptable terminations, False otherwise.

        Parameters
        ----------
        acceptable_terminations : set of pyomo.opt.TerminationCondition
            Acceptable termination conditions.

        Returns
        -------
        bool
        """
        return any((res.solver.termination_condition in acceptable_terminations for res in self.results_list))

    def evaluate_total_solve_time(self, evaluator_func, **evaluator_func_kwargs):
        """
        Evaluate total time required by subordinate solvers
        for separation problem of interest, according to Pyomo
        ``SolverResults`` objects stored in ``self.results_list``.

        Parameters
        ----------
        evaluator_func : callable
            Solve time evaluator function.
            This callable should accept an object of type
            ``pyomo.opt.results.SolverResults``, and
            return a float equal to the time required.
        **evaluator_func_kwargs : dict, optional
            Keyword arguments to evaluator function.

        Returns
        -------
        float
            Total time spent by solvers.
        """
        return sum((evaluator_func(res, **evaluator_func_kwargs) for res in self.results_list))