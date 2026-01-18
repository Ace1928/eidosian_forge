import warnings
import cvxpy.error as error
import cvxpy.problems as problems
import cvxpy.settings as s
from cvxpy.problems.objective import Minimize
from cvxpy.reductions.solution import failure_solution
def _find_bisection_interval(problem, t, solver=None, low=None, high=None, max_iters=100):
    """Finds an interval for bisection."""
    if low is None:
        low = 0 if t.is_nonneg() else -1
    if high is None:
        high = 0 if t.is_nonpos() else 1
    infeasible_low = t.is_nonneg()
    feasible_high = t.is_nonpos()
    for _ in range(max_iters):
        if not feasible_high:
            t.value = high
            lowered = _lower_problem(problem)
            _solve(lowered, solver)
            if _infeasible(lowered):
                low = high
                high *= 2
                continue
            elif lowered.status in s.SOLUTION_PRESENT:
                feasible_high = True
            else:
                raise error.SolverError('Solver failed with status %s' % lowered.status)
        if not infeasible_low:
            t.value = low
            lowered = _lower_problem(problem)
            _solve(lowered, solver=solver)
            if _infeasible(lowered):
                infeasible_low = True
            elif lowered.status in s.SOLUTION_PRESENT:
                high = low
                low *= 2
                continue
            else:
                raise error.SolverError('Solver failed with status %s' % lowered.status)
        if infeasible_low and feasible_high:
            return (low, high)
    raise error.SolverError('Unable to find suitable interval for bisection; your problem may be unbounded..')