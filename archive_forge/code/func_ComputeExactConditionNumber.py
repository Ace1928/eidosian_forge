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
def ComputeExactConditionNumber(self):
    """
         Advanced usage: computes the exact condition number of the current scaled
        basis: L1norm(B) * L1norm(inverse(B)), where B is the scaled basis.

        This method requires that a basis exists: it should be called after Solve.
        It is only available for continuous problems. It is implemented for GLPK
        but not CLP because CLP does not provide the API for doing it.

        The condition number measures how well the constraint matrix is conditioned
        and can be used to predict whether numerical issues will arise during the
        solve: the model is declared infeasible whereas it is feasible (or
        vice-versa), the solution obtained is not optimal or violates some
        constraints, the resolution is slow because of repeated singularities.

        The rule of thumb to interpret the condition number kappa is:
          - o kappa <= 1e7: virtually no chance of numerical issues
          - o 1e7 < kappa <= 1e10: small chance of numerical issues
          - o 1e10 < kappa <= 1e13: medium chance of numerical issues
          - o kappa > 1e13: high chance of numerical issues

        The computation of the condition number depends on the quality of the LU
        decomposition, so it is not very accurate when the matrix is ill
        conditioned.
        """
    return _pywraplp.Solver_ComputeExactConditionNumber(self)