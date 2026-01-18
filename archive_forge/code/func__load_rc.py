import logging
import re
import sys
import itertools
import operator
import pyomo.core.base.var
import pyomo.core.base.constraint
from pyomo.common.dependencies import attempt_import
from pyomo.common.tempfiles import TempfileManager
from pyomo.core import is_fixed, value, minimize, maximize
from pyomo.core.base.suffix import Suffix
from pyomo.core.staleflag import StaleFlagManager
from pyomo.opt.base.solvers import OptSolver
from pyomo.repn import generate_standard_repn
from pyomo.solvers.plugins.solvers.direct_solver import DirectSolver
from pyomo.solvers.plugins.solvers.direct_or_persistent_solver import (
from pyomo.common.collections import ComponentMap, ComponentSet, Bunch
from pyomo.opt import SolverFactory
from pyomo.core.kernel.conic import (
from pyomo.opt.results.results_ import SolverResults
from pyomo.opt.results.solution import Solution, SolutionStatus
from pyomo.opt.results.solver import TerminationCondition, SolverStatus
def _load_rc(self, vars_to_load=None):
    if not hasattr(self._pyomo_model, 'rc'):
        self._pyomo_model.rc = Suffix(direction=Suffix.IMPORT)
    var_map = self._pyomo_var_to_solver_var_map
    ref_vars = self._referenced_variables
    rc = self._pyomo_model.rc
    if vars_to_load is None:
        vars_to_load = var_map.keys()
    mosek_vars_to_load = [var_map[pyomo_var] for pyomo_var in vars_to_load]
    vals = [0.0] * len(mosek_vars_to_load)
    self._solver_model.getreducedcosts(self._whichsol, 0, len(mosek_vars_to_load), vals)
    for var, val in zip(vars_to_load, vals):
        if ref_vars[var] > 0:
            rc[var] = val