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
def _process_domain_and_bounds(self, var, var_id, mutable_lbs, mutable_ubs, ndx, gurobipy_var):
    _v, _lb, _ub, _fixed, _domain_interval, _value = self._vars[id(var)]
    lb, ub, step = _domain_interval
    if lb is None:
        lb = -gurobipy.GRB.INFINITY
    if ub is None:
        ub = gurobipy.GRB.INFINITY
    if step == 0:
        vtype = gurobipy.GRB.CONTINUOUS
    elif step == 1:
        if lb == 0 and ub == 1:
            vtype = gurobipy.GRB.BINARY
        else:
            vtype = gurobipy.GRB.INTEGER
    else:
        raise ValueError(f'Unrecognized domain step: {step} (should be either 0 or 1)')
    if _fixed:
        lb = _value
        ub = _value
    else:
        if _lb is not None:
            if not is_constant(_lb):
                mutable_bound = _MutableLowerBound(NPV_MaxExpression((_lb, lb)))
                if gurobipy_var is None:
                    mutable_lbs[ndx] = mutable_bound
                else:
                    mutable_bound.var = gurobipy_var
                self._mutable_bounds[var_id, 'lb'] = (var, mutable_bound)
            lb = max(value(_lb), lb)
        if _ub is not None:
            if not is_constant(_ub):
                mutable_bound = _MutableUpperBound(NPV_MinExpression((_ub, ub)))
                if gurobipy_var is None:
                    mutable_ubs[ndx] = mutable_bound
                else:
                    mutable_bound.var = gurobipy_var
                self._mutable_bounds[var_id, 'ub'] = (var, mutable_bound)
            ub = min(value(_ub), ub)
    return (lb, ub, vtype)