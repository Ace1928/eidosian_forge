from pyomo.core.expr.numvalue import value
from pyomo.solvers.plugins.solvers.cplex_direct import CPLEXDirect
from pyomo.solvers.plugins.solvers.persistent_solver import PersistentSolver
from pyomo.opt.base import SolverFactory
def _remove_constraint(self, solver_con):
    try:
        self._solver_model.linear_constraints.delete(solver_con)
    except self._cplex.exceptions.CplexError:
        try:
            self._solver_model.quadratic_constraints.delete(solver_con)
        except self._cplex.exceptions.CplexError:
            raise ValueError('Failed to find the cplex constraint {0}'.format(solver_con))