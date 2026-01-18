from pyomo.core.expr.numvalue import value
from pyomo.solvers.plugins.solvers.cplex_direct import CPLEXDirect
from pyomo.solvers.plugins.solvers.persistent_solver import PersistentSolver
from pyomo.opt.base import SolverFactory
def _remove_sos_constraint(self, solver_sos_con):
    self._solver_model.SOS.delete(solver_sos_con)