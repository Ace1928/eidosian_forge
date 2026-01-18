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
class DecomposedImplicitFunctionBase(PyomoImplicitFunctionBase):
    """A base class for an implicit function that applies a partition
    to its variables and constraints and converges the system by solving
    subsets sequentially

    Subclasses should implement the partition_system method, which
    determines how variables and constraints are partitioned into subsets.

    """

    def __init__(self, variables, constraints, parameters, solver_class=None, solver_options=None, timer=None, use_calc_var=True):
        if timer is None:
            timer = HierarchicalTimer()
        self._timer = timer
        self._timer.start('__init__')
        if solver_class is None:
            solver_class = ScipySolverWrapper
        self._solver_class = solver_class
        if solver_options is None:
            solver_options = {}
        self._solver_options = solver_options
        self._calc_var_cutoff = 1 if use_calc_var else 0
        super().__init__(variables, constraints, parameters)
        subsystem_list = [(cons, vars) for vars, cons in self.partition_system(variables, constraints)]
        var_param_set = ComponentSet(variables + parameters)
        constants = []
        constant_set = ComponentSet()
        for con in constraints:
            for var in identify_variables(con.expr, include_fixed=False):
                if var not in constant_set and var not in var_param_set:
                    constant_set.add(var)
                    constants.append(var)
        with TemporarySubsystemManager(to_fix=constants):
            self._subsystem_list = list(generate_subsystem_blocks(subsystem_list))
            self._solver_subsystem_list = [(block, inputs) for block, inputs in self._subsystem_list if len(block.vars) > self._calc_var_cutoff]
            for block, inputs in self._solver_subsystem_list:
                block._obj = Objective(expr=0.0)
                block.scaling_factor = Suffix(direction=Suffix.EXPORT)
                block.scaling_factor[block._obj] = 1.0
            self._timer.start('PyomoNLP')
            self._solver_subsystem_nlps = [pyomo_nlp.PyomoNLP(block) for block, inputs in self._solver_subsystem_list]
            self._timer.stop('PyomoNLP')
        self._solver_subsystem_var_names = [[var.name for var in block.vars.values()] for block, inputs in self._solver_subsystem_list]
        self._solver_proj_nlps = [nlp_proj.ProjectedExtendedNLP(nlp, names) for nlp, names in zip(self._solver_subsystem_nlps, self._solver_subsystem_var_names)]
        self._timer.start('NlpSolver')
        self._nlp_solvers = [self._solver_class(nlp, timer=self._timer, options=self._solver_options) for nlp in self._solver_proj_nlps]
        self._timer.stop('NlpSolver')
        self._solver_subsystem_input_coords = [nlp.get_primal_indices(inputs) for nlp, (subsystem, inputs) in zip(self._solver_subsystem_nlps, self._solver_subsystem_list)]
        self._n_variables = len(variables)
        self._n_constraints = len(constraints)
        self._n_parameters = len(parameters)
        self._global_values = np.array([var.value for var in variables + parameters])
        self._global_indices = ComponentMap(((var, i) for i, var in enumerate(variables + parameters)))
        self._local_input_global_coords = [np.array([self._global_indices[var] for var in inputs], dtype=int) for _, inputs in self._solver_subsystem_list]
        self._output_coords = [np.array([self._global_indices[var] for var in block.vars.values()], dtype=int) for block, _ in self._solver_subsystem_list]
        self._timer.stop('__init__')

    def n_subsystems(self):
        """Returns the number of subsystems in the partition of variables
        and equations used to converge the system defining the implicit
        function

        """
        return len(self._subsystem_list)

    def partition_system(self, variables, constraints):
        """Partitions the systems of equations defined by the provided
        variables and constraints

        Each subset of the partition should have an equal number of variables
        and equations. These subsets, or "subsystems", will be solved
        sequentially in the order provided by this method instead of solving
        the entire system simultaneously. Subclasses should implement this
        method to define the partition that their implicit function solver
        will use. Partitions are defined as a list of tuples of lists.
        Each tuple has two entries, the first a list of variables, and the
        second a list of constraints. These inner lists should have the
        same number of entries.

        Arguments
        ---------
        variables: list
            List of VarData in the system to be partitioned
        constraints: list
            List of ConstraintData (equality constraints) defining the
            equations of the system to be partitioned

        Returns
        -------
        List of tuples
            List of tuples describing the ordered partition. Each tuple
            contains equal-length subsets of variables and constraints.

        """
        raise NotImplementedError('%s has not implemented the partition_system method' % type(self))

    def set_parameters(self, values):
        self._timer.start('set_parameters')
        values = np.array(values)
        self._global_values[self._n_variables:] = values
        solver_subsystem_idx = 0
        for block, inputs in self._subsystem_list:
            if len(block.vars) <= self._calc_var_cutoff:
                for var in inputs:
                    idx = self._global_indices[var]
                    var.set_value(self._global_values[idx], skip_validation=True)
                var = block.vars[0]
                con = block.cons[0]
                self._timer.start('solve')
                self._timer.start('calc_var')
                calculate_variable_from_constraint(var, con)
                self._timer.stop('calc_var')
                self._timer.stop('solve')
                self._global_values[self._global_indices[var]] = var.value
            else:
                i = solver_subsystem_idx
                nlp = self._solver_subsystem_nlps[i]
                proj_nlp = self._solver_proj_nlps[i]
                input_coords = self._solver_subsystem_input_coords[i]
                input_global_coords = self._local_input_global_coords[i]
                output_global_coords = self._output_coords[i]
                nlp_solver = self._nlp_solvers[solver_subsystem_idx]
                primals = nlp.get_primals()
                primals[input_coords] = self._global_values[input_global_coords]
                nlp.set_primals(primals)
                x0 = proj_nlp.get_primals()
                self._timer.start('solve')
                self._timer.start('solve_nlp')
                nlp_solver.solve(x0=x0)
                self._timer.stop('solve_nlp')
                self._timer.stop('solve')
                self._global_values[output_global_coords] = proj_nlp.get_primals()
                solver_subsystem_idx += 1
        self._timer.stop('set_parameters')

    def evaluate_outputs(self):
        return self._global_values[:self._n_variables]

    def update_pyomo_model(self):
        for i, var in enumerate(self.get_variables() + self.get_parameters()):
            var.set_value(self._global_values[i], skip_validation=True)