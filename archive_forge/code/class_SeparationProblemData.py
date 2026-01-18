class SeparationProblemData(object):
    """
    Container for the grcs separation problem

    Attributes:
        :separation_model: separation problem model object
        :points_added_to_master: list of parameter violations added to the master problem over the course of the algorithm
        :separation_problem_subsolver_statuses: list of subordinate sub-solver statuses throughout separations
        :total_global_separation_solvers: Counter for number of times global solvers were employed in separation
        :constraint_violations: List of constraint violations identified in separation
    """
    pass