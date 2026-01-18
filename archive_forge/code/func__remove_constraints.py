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
def _remove_constraints(self, cons: List[_GeneralConstraintData]):
    for con in cons:
        if con in self._constraints_added_since_update:
            self._update_gurobi_model()
        solver_con = self._pyomo_con_to_solver_con_map[con]
        self._solver_model.remove(solver_con)
        self._symbol_map.removeSymbol(con)
        del self._pyomo_con_to_solver_con_map[con]
        del self._solver_con_to_pyomo_con_map[id(solver_con)]
        self._range_constraints.discard(con)
        self._mutable_helpers.pop(con, None)
        self._mutable_quadratic_helpers.pop(con, None)
    self._needs_updated = True