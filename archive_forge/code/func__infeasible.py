import warnings
import cvxpy.error as error
import cvxpy.problems as problems
import cvxpy.settings as s
from cvxpy.problems.objective import Minimize
from cvxpy.reductions.solution import failure_solution
def _infeasible(problem) -> bool:
    return problem is None or problem.status in (s.INFEASIBLE, s.INFEASIBLE_INACCURATE)