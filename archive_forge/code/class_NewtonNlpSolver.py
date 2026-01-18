from collections import namedtuple
from pyomo.core.base.objective import Objective
from pyomo.common.timing import HierarchicalTimer
from pyomo.common.modeling import unique_component_name
from pyomo.common.config import ConfigBlock, ConfigValue, In
from pyomo.contrib.pynumero.algorithms.solvers.square_solver_base import (
from pyomo.opt import SolverResults, TerminationCondition
from pyomo.common.dependencies import (
class NewtonNlpSolver(ScalarDenseSquareNlpSolver):
    """A wrapper around the SciPy scalar Newton solver for NLP objects"""
    OPTIONS = ScalarDenseSquareNlpSolver.OPTIONS(description='Options for SciPy newton wrapper')
    OPTIONS.declare('tol', ConfigValue(default=1e-08, domain=float, description='Convergence tolerance'))
    OPTIONS.declare('secant', ConfigValue(default=False, domain=bool, description="Whether to use SciPy's secant method"))
    OPTIONS.declare('full_output', ConfigValue(default=True, domain=bool, description='Whether underlying solver should return its full output'))
    OPTIONS.declare('maxiter', ConfigValue(default=50, domain=int, description='Maximum number of function evaluations per solve'))

    def solve(self, x0=None):
        if x0 is None:
            x0 = self._nlp.get_primals()
        if self.options.secant:
            fprime = None
        else:
            fprime = lambda x: self.evaluate_jacobian(np.array([x]))[0, 0]
        results = sp.optimize.newton(lambda x: self.evaluate_function(np.array([x]))[0], x0[0], fprime=fprime, tol=self.options.tol, full_output=self.options.full_output, maxiter=self.options.maxiter)
        return results