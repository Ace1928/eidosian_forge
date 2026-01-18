from collections import namedtuple
from pyomo.core.base.objective import Objective
from pyomo.common.timing import HierarchicalTimer
from pyomo.common.modeling import unique_component_name
from pyomo.common.config import ConfigBlock, ConfigValue, In
from pyomo.contrib.pynumero.algorithms.solvers.square_solver_base import (
from pyomo.opt import SolverResults, TerminationCondition
from pyomo.common.dependencies import (
class PyomoFsolveSolver(PyomoScipySolver):
    _term_cond = {1: TerminationCondition.feasible}

    def create_nlp_solver(self, **kwds):
        nlp = self.get_nlp()
        solver = FsolveNlpSolver(nlp, **kwds)
        return solver

    def get_pyomo_results(self, model, scipy_results):
        nlp = self.get_nlp()
        if self._nlp_solver.options.full_output:
            x, info, ier, msg = scipy_results
        else:
            x, ier, msg = scipy_results
        results = SolverResults()
        results.problem.name = model.name
        results.problem.number_of_constraints = nlp.n_eq_constraints()
        results.problem.number_of_variables = nlp.n_primals()
        results.problem.number_of_binary_variables = 0
        results.problem.number_of_integer_variables = 0
        results.problem.number_of_continuous_variables = nlp.n_primals()
        results.solver.name = 'scipy.fsolve'
        results.solver.return_code = ier
        results.solver.message = msg
        results.solver.wallclock_time = self._timer.timers['solve'].total_time
        results.solver.termination_condition = self._term_cond.get(ier, TerminationCondition.error)
        results.solver.status = TerminationCondition.to_solver_status(results.solver.termination_condition)
        if self._nlp_solver.options.full_output:
            results.solver.number_of_function_evaluations = info['nfev']
            results.solver.number_of_gradient_evaluations = info['njev']
        return results