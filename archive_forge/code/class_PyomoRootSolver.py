from collections import namedtuple
from pyomo.core.base.objective import Objective
from pyomo.common.timing import HierarchicalTimer
from pyomo.common.modeling import unique_component_name
from pyomo.common.config import ConfigBlock, ConfigValue, In
from pyomo.contrib.pynumero.algorithms.solvers.square_solver_base import (
from pyomo.opt import SolverResults, TerminationCondition
from pyomo.common.dependencies import (
class PyomoRootSolver(PyomoScipySolver):

    def create_nlp_solver(self, **kwds):
        nlp = self.get_nlp()
        solver = RootNlpSolver(nlp, **kwds)
        return solver

    def get_pyomo_results(self, model, scipy_results):
        nlp = self.get_nlp()
        results = SolverResults()
        results.problem.name = model.name
        results.problem.number_of_constraints = nlp.n_eq_constraints()
        results.problem.number_of_variables = nlp.n_primals()
        results.problem.number_of_binary_variables = 0
        results.problem.number_of_integer_variables = 0
        results.problem.number_of_continuous_variables = nlp.n_primals()
        results.solver.name = 'scipy.root'
        results.solver.return_code = scipy_results.status
        results.solver.message = scipy_results.message
        results.solver.wallclock_time = self._timer.timers['solve'].total_time
        if scipy_results.success:
            results.solver.termination_condition = TerminationCondition.feasible
        else:
            results.solver.termination_condition = TerminationCondition.error
        results.solver.status = TerminationCondition.to_solver_status(results.solver.termination_condition)
        results.solver.number_of_function_evaluations = scipy_results.nfev
        results.solver.number_of_gradient_evaluations = scipy_results.njev
        return results