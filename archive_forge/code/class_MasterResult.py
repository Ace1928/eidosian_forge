class MasterResult(object):
    """Data class for master problem results data.

    Attributes:
         - termination_condition: Solver termination condition
         - fsv_values: list of design variable values
         - ssv_values: list of control variable values
         - first_stage_objective: objective contribution due to first-stage degrees of freedom
         - second_stage_objective: objective contribution due to second-stage degrees of freedom
         - grcs_termination_condition: the conditions under which the grcs terminated
                                       (max_iter, robust_optimal, error)
         - pyomo_results: results object from solve() statement

    """