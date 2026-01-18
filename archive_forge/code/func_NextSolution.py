from sys import version_info as _swig_python_version_info
import numbers
from ortools.linear_solver.python.linear_solver_natural_api import OFFSET_KEY
from ortools.linear_solver.python.linear_solver_natural_api import inf
from ortools.linear_solver.python.linear_solver_natural_api import LinearExpr
from ortools.linear_solver.python.linear_solver_natural_api import ProductCst
from ortools.linear_solver.python.linear_solver_natural_api import Sum
from ortools.linear_solver.python.linear_solver_natural_api import SumArray
from ortools.linear_solver.python.linear_solver_natural_api import SumCst
from ortools.linear_solver.python.linear_solver_natural_api import LinearConstraint
from ortools.linear_solver.python.linear_solver_natural_api import VariableExpr
def NextSolution(self):
    """
        Some solvers (MIP only, not LP) can produce multiple solutions to the
        problem. Returns true when another solution is available, and updates the
        MPVariable* objects to make the new solution queryable. Call only after
        calling solve.

        The optimality properties of the additional solutions found, and whether or
        not the solver computes them ahead of time or when NextSolution() is called
        is solver specific.

        As of 2020-02-10, only Gurobi and SCIP support NextSolution(), see
        linear_solver_interfaces_test for an example of how to configure these
        solvers for multiple solutions. Other solvers return false unconditionally.
        """
    return _pywraplp.Solver_NextSolution(self)