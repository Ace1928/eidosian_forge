import logging
import re
import sys
from pyomo.common.collections import ComponentSet, ComponentMap, Bunch
from pyomo.common.dependencies import attempt_import
from pyomo.common.errors import ApplicationError
from pyomo.common.tempfiles import TempfileManager
from pyomo.common.tee import capture_output
from pyomo.core.expr.numvalue import is_fixed
from pyomo.core.expr.numvalue import value
from pyomo.core.staleflag import StaleFlagManager
from pyomo.repn import generate_standard_repn
from pyomo.solvers.plugins.solvers.direct_solver import DirectSolver
from pyomo.solvers.plugins.solvers.direct_or_persistent_solver import (
from pyomo.core.kernel.objective import minimize, maximize
from pyomo.opt.results.results_ import SolverResults
from pyomo.opt.results.solution import Solution, SolutionStatus
from pyomo.opt.results.solver import TerminationCondition, SolverStatus
from pyomo.opt.base import SolverFactory
from pyomo.core.base.suffix import Suffix
import pyomo.core.base.var
def _add_sos_constraint(self, con):
    if not con.active:
        return None
    conname = self._symbol_map.getSymbol(con, self._labeler)
    level = con.level
    if level == 1:
        sos_type = gurobipy.GRB.SOS_TYPE1
    elif level == 2:
        sos_type = gurobipy.GRB.SOS_TYPE2
    else:
        raise ValueError('Solver does not support SOS level {0} constraints'.format(level))
    gurobi_vars = []
    weights = []
    self._vars_referenced_by_con[con] = ComponentSet()
    if hasattr(con, 'get_items'):
        sos_items = list(con.get_items())
    else:
        sos_items = list(con.items())
    for v, w in sos_items:
        self._vars_referenced_by_con[con].add(v)
        gurobi_vars.append(self._pyomo_var_to_solver_var_map[v])
        self._referenced_variables[v] += 1
        weights.append(w)
    gurobipy_con = self._solver_model.addSOS(sos_type, gurobi_vars, weights)
    self._pyomo_con_to_solver_con_map[con] = gurobipy_con
    self._solver_con_to_pyomo_con_map[gurobipy_con] = con
    self._needs_updated = True