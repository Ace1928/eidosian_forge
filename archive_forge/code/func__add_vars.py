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
def _add_vars(self, var_seq):
    if not var_seq:
        return
    var_num = self._solver_model.getnumvar()
    vnames = tuple((self._symbol_map.getSymbol(v, self._labeler) for v in var_seq))
    vtypes = tuple(map(self._mosek_vartype_from_var, var_seq))
    lbs = tuple((value(v) if v.fixed else -inf if value(v.lb) is None else value(v.lb) for v in var_seq))
    ubs = tuple((value(v) if v.fixed else inf if value(v.ub) is None else value(v.ub) for v in var_seq))
    fxs = tuple((v.is_fixed() for v in var_seq))
    bound_types = tuple(map(self._mosek_bounds, lbs, ubs, fxs))
    self._solver_model.appendvars(len(var_seq))
    var_ids = range(var_num, var_num + len(var_seq))
    _vnames = tuple(map(self._solver_model.putvarname, var_ids, vnames))
    self._solver_model.putvartypelist(var_ids, vtypes)
    self._solver_model.putvarboundlist(var_ids, bound_types, lbs, ubs)
    self._pyomo_var_to_solver_var_map.update(zip(var_seq, var_ids))
    self._solver_var_to_pyomo_var_map.update(zip(var_ids, var_seq))
    self._referenced_variables.update(zip(var_seq, [0] * len(var_seq)))