from pyomo.core.expr.numvalue import value
from pyomo.solvers.plugins.solvers.cplex_direct import CPLEXDirect
from pyomo.solvers.plugins.solvers.persistent_solver import PersistentSolver
from pyomo.opt.base import SolverFactory
def _remove_var(self, solver_var):
    pyomo_var = self._solver_var_to_pyomo_var_map[solver_var]
    ndx = self._pyomo_var_to_ndx_map[pyomo_var]
    for tmp_var, tmp_ndx in self._pyomo_var_to_ndx_map.items():
        if tmp_ndx > ndx:
            self._pyomo_var_to_ndx_map[tmp_var] -= 1
    self._ndx_count -= 1
    del self._pyomo_var_to_ndx_map[pyomo_var]
    self._solver_model.variables.delete(solver_var)