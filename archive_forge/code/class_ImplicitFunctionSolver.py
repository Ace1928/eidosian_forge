from pyomo.common.collections import ComponentSet, ComponentMap
from pyomo.common.timing import HierarchicalTimer
from pyomo.common.dependencies import attempt_import, numpy as np
from pyomo.core.base.objective import Objective
from pyomo.core.base.suffix import Suffix
from pyomo.core.expr.visitor import identify_variables
from pyomo.util.calc_var_value import calculate_variable_from_constraint
from pyomo.util.subsystems import (
from pyomo.contrib.pynumero.algorithms.solvers.cyipopt_solver import CyIpoptSolver
from pyomo.contrib.pynumero.interfaces.cyipopt_interface import CyIpoptNLP
from pyomo.contrib.pynumero.algorithms.solvers.scipy_solvers import (
from pyomo.contrib.incidence_analysis import IncidenceGraphInterface
from pyomo.contrib.incidence_analysis.scc_solver import (
class ImplicitFunctionSolver(PyomoImplicitFunctionBase):
    """A basic implicit function solver that uses a ProjectedNLP to solve
    the parameterized system without repeated file writes when parameters
    are updated

    """

    def __init__(self, variables, constraints, parameters, solver_class=None, solver_options=None, timer=None):
        if timer is None:
            timer = HierarchicalTimer()
        self._timer = timer
        if solver_options is None:
            solver_options = {}
        self._timer.start('__init__')
        super().__init__(variables, constraints, parameters)
        block = self.get_block()
        block._obj = Objective(expr=0.0)
        block.scaling_factor = Suffix(direction=Suffix.EXPORT)
        block.scaling_factor[block._obj] = 1.0
        self._timer.start('PyomoNLP')
        self._nlp = pyomo_nlp.PyomoNLP(block)
        self._timer.stop('PyomoNLP')
        primals_ordering = [var.name for var in variables]
        self._proj_nlp = nlp_proj.ProjectedExtendedNLP(self._nlp, primals_ordering)
        self._timer.start('NlpSolver')
        if solver_class is None:
            self._solver = ScipySolverWrapper(self._proj_nlp, options=solver_options, timer=timer)
        else:
            self._solver = solver_class(self._proj_nlp, options=solver_options, timer=timer)
        self._timer.stop('NlpSolver')
        vars_in_cons = []
        _seen = set()
        for con in constraints:
            for var in identify_variables(con.expr, include_fixed=False):
                if id(var) not in _seen:
                    _seen.add(id(var))
                    vars_in_cons.append(var)
        self._active_var_set = ComponentSet(vars_in_cons)
        self._active_param_mask = np.array([p in self._active_var_set for p in parameters])
        self._active_parameters = [p for i, p in enumerate(parameters) if self._active_param_mask[i]]
        if any((var not in self._active_var_set for var in variables)):
            raise RuntimeError('Invalid model. All variables must appear in specified constraints.')
        self._variable_coords = self._nlp.get_primal_indices(variables)
        self._active_parameter_coords = self._nlp.get_primal_indices(self._active_parameters)
        self._parameter_values = np.array([var.value for var in parameters])
        self._timer.start('__init__')

    def set_parameters(self, values, **kwds):
        self._timer.start('set_parameters')
        values = np.array(values)
        self._parameter_values = values
        values = np.compress(self._active_param_mask, values)
        primals = self._nlp.get_primals()
        primals[self._active_parameter_coords] = values
        self._nlp.set_primals(primals)
        self._timer.start('solve')
        results = self._solver.solve(**kwds)
        self._timer.stop('solve')
        self._timer.stop('set_parameters')
        return results

    def evaluate_outputs(self):
        primals = self._nlp.get_primals()
        outputs = primals[self._variable_coords]
        return outputs

    def update_pyomo_model(self):
        primals = self._nlp.get_primals()
        for i, var in enumerate(self.get_variables()):
            var.set_value(primals[self._variable_coords[i]], skip_validation=True)
        for var, value in zip(self._parameters, self._parameter_values):
            var.set_value(value, skip_validation=True)