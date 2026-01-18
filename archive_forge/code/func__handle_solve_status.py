from collections import namedtuple
from operator import attrgetter
import numpy as np
from scipy.sparse import dok_matrix
import cvxpy.settings as s
from cvxpy.constraints import SOC
from cvxpy.reductions.solution import Solution, failure_solution
from cvxpy.reductions.solvers import utilities
from cvxpy.reductions.solvers.conic_solvers.conic_solver import (
def _handle_solve_status(model, solstat):
    """Map CPLEX MIP solution status codes to non-MIP status codes."""
    status = model.solution.status
    if model.get_versionnumber() < 12100000:
        unbounded_status_codes = (status.MIP_unbounded, status.MIP_benders_master_unbounded, status.benders_master_unbounded)
    else:
        unbounded_status_codes = (status.MIP_unbounded,)
    if solstat == status.MIP_optimal:
        return status.optimal
    elif solstat == status.MIP_infeasible:
        return status.infeasible
    elif solstat in (status.MIP_time_limit_feasible, status.MIP_time_limit_infeasible):
        return status.abort_time_limit
    elif solstat in (status.MIP_dettime_limit_feasible, status.MIP_dettime_limit_infeasible):
        return status.abort_dettime_limit
    elif solstat in (status.MIP_abort_feasible, status.MIP_abort_infeasible):
        return status.abort_user
    elif solstat == status.MIP_optimal_infeasible:
        return status.optimal_infeasible
    elif solstat == status.MIP_infeasible_or_unbounded:
        return status.infeasible_or_unbounded
    elif solstat in unbounded_status_codes:
        return status.unbounded
    elif solstat in (status.feasible_relaxed_sum, status.MIP_feasible_relaxed_sum, status.optimal_relaxed_sum, status.MIP_optimal_relaxed_sum, status.feasible_relaxed_inf, status.MIP_feasible_relaxed_inf, status.optimal_relaxed_inf, status.MIP_optimal_relaxed_inf, status.feasible_relaxed_quad, status.MIP_feasible_relaxed_quad, status.optimal_relaxed_quad, status.MIP_optimal_relaxed_quad):
        raise AssertionError('feasopt status encountered: {0}'.format(solstat))
    elif solstat in (status.conflict_feasible, status.conflict_minimal, status.conflict_abort_contradiction, status.conflict_abort_time_limit, status.conflict_abort_dettime_limit, status.conflict_abort_iteration_limit, status.conflict_abort_node_limit, status.conflict_abort_obj_limit, status.conflict_abort_memory_limit, status.conflict_abort_user):
        raise AssertionError('conflict refiner status encountered: {0}'.format(solstat))
    elif solstat == status.relaxation_unbounded:
        return status.relaxation_unbounded
    elif solstat in (status.feasible, status.MIP_feasible):
        return status.feasible
    elif solstat == status.benders_num_best:
        return status.num_best
    else:
        return solstat