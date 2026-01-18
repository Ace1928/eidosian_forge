from collections.abc import Iterable
import logging
import math
from typing import List, Optional
from pyomo.common.collections import ComponentSet, ComponentMap, OrderedSet
from pyomo.common.dependencies import attempt_import
from pyomo.common.errors import PyomoException
from pyomo.common.tee import capture_output, TeeStream
from pyomo.common.timing import HierarchicalTimer
from pyomo.common.shutdown import python_is_shutting_down
from pyomo.common.config import ConfigValue
from pyomo.core.kernel.objective import minimize, maximize
from pyomo.core.base import SymbolMap, NumericLabeler, TextLabeler
from pyomo.core.base.var import _GeneralVarData
from pyomo.core.base.constraint import _GeneralConstraintData
from pyomo.core.base.sos import _SOSConstraintData
from pyomo.core.base.param import _ParamData
from pyomo.core.expr.numvalue import value, is_constant, is_fixed, native_numeric_types
from pyomo.repn import generate_standard_repn
from pyomo.core.expr.numeric_expr import NPV_MaxExpression, NPV_MinExpression
from pyomo.contrib.solver.base import PersistentSolverBase
from pyomo.contrib.solver.results import Results, TerminationCondition, SolutionStatus
from pyomo.contrib.solver.config import PersistentBranchAndBoundConfig
from pyomo.contrib.solver.persistent import PersistentSolverUtils
from pyomo.contrib.solver.solution import PersistentSolutionLoader
from pyomo.core.staleflag import StaleFlagManager
import sys
import datetime
import io
def _get_reduced_costs(self, vars_to_load=None):
    if self._needs_updated:
        self._update_gurobi_model()
    if self._solver_model.Status != gurobipy.GRB.OPTIMAL:
        raise RuntimeError('Solver does not currently have valid reduced costs. Please check the termination condition.')
    var_map = self._pyomo_var_to_solver_var_map
    ref_vars = self._referenced_variables
    res = ComponentMap()
    if vars_to_load is None:
        vars_to_load = self._pyomo_var_to_solver_var_map.keys()
    else:
        vars_to_load = [id(v) for v in vars_to_load]
    gurobi_vars_to_load = [var_map[pyomo_var_id] for pyomo_var_id in vars_to_load]
    vals = self._solver_model.getAttr('Rc', gurobi_vars_to_load)
    for var_id, val in zip(vars_to_load, vals):
        using_cons, using_sos, using_obj = ref_vars[var_id]
        if using_cons or using_sos or using_obj is not None:
            res[self._vars[var_id][0]] = val
    return res