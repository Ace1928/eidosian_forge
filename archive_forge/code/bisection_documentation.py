import warnings
import cvxpy.error as error
import cvxpy.problems as problems
import cvxpy.settings as s
from cvxpy.problems.objective import Minimize
from cvxpy.reductions.solution import failure_solution
Bisection on a one-parameter family of DCP problems.

    Bisects on a one-parameter family of DCP problems emitted by `Dqcp2Dcp`.

    Parameters
    ------
    problem : Problem
        problem emitted by Dqcp2Dcp
    solver : Solver
        solver to use for bisection
    low : float
        lower bound for bisection (optional)
    high : float
        upper bound for bisection (optional)
    eps : float
        terminate bisection when width of interval is < eps
    verbose : bool
        whether to print verbose output related to the bisection
    max_iters : int
        the maximum number of iterations to run bisection

    Returns
    -------
    A Solution object.
    