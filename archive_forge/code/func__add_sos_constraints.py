from collections.abc import Iterable
import logging
import math
from typing import List, Dict, Optional
from pyomo.common.collections import ComponentSet, ComponentMap, OrderedSet
from pyomo.common.log import LogStream
from pyomo.common.dependencies import attempt_import
from pyomo.common.errors import PyomoException
from pyomo.common.tee import capture_output, TeeStream
from pyomo.common.timing import HierarchicalTimer
from pyomo.common.shutdown import python_is_shutting_down
from pyomo.common.config import ConfigValue, NonNegativeInt
from pyomo.core.kernel.objective import minimize, maximize
from pyomo.core.base import SymbolMap, NumericLabeler, TextLabeler
from pyomo.core.base.var import Var, _GeneralVarData
from pyomo.core.base.constraint import _GeneralConstraintData
from pyomo.core.base.sos import _SOSConstraintData
from pyomo.core.base.param import _ParamData
from pyomo.core.expr.numvalue import value, is_constant, is_fixed, native_numeric_types
from pyomo.repn import generate_standard_repn
from pyomo.core.expr.numeric_expr import NPV_MaxExpression, NPV_MinExpression
from pyomo.contrib.appsi.base import (
from pyomo.contrib.appsi.cmodel import cmodel, cmodel_available
from pyomo.core.staleflag import StaleFlagManager
import sys
def _add_sos_constraints(self, cons: List[_SOSConstraintData]):
    for con in cons:
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
        for v, w in con.get_items():
            v_id = id(v)
            gurobi_vars.append(self._pyomo_var_to_solver_var_map[v_id])
            weights.append(w)
        gurobipy_con = self._solver_model.addSOS(sos_type, gurobi_vars, weights)
        self._pyomo_sos_to_solver_sos_map[con] = gurobipy_con
    self._constraints_added_since_update.update(cons)
    self._needs_updated = True