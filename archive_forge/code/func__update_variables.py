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
def _update_variables(self, variables: List[_GeneralVarData]):
    for var in variables:
        var_id = id(var)
        if var_id not in self._pyomo_var_to_solver_var_map:
            raise ValueError('The Var provided to update_var needs to be added first: {0}'.format(var))
        self._mutable_bounds.pop((var_id, 'lb'), None)
        self._mutable_bounds.pop((var_id, 'ub'), None)
        gurobipy_var = self._pyomo_var_to_solver_var_map[var_id]
        lb, ub, vtype = self._process_domain_and_bounds(var, var_id, None, None, None, gurobipy_var)
        gurobipy_var.setAttr('lb', lb)
        gurobipy_var.setAttr('ub', ub)
        gurobipy_var.setAttr('vtype', vtype)
    self._needs_updated = True