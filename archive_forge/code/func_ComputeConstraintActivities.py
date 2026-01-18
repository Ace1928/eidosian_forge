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
def ComputeConstraintActivities(self):
    """
        Advanced usage: compute the "activities" of all constraints, which are the
        sums of their linear terms. The activities are returned in the same order
        as constraints(), which is the order in which constraints were added; but
        you can also use MPConstraint::index() to get a constraint's index.
        """
    return _pywraplp.Solver_ComputeConstraintActivities(self)