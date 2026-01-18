import logging
from typing import List, Dict, Optional
from pyomo.common.collections import ComponentMap
from pyomo.common.dependencies import attempt_import
from pyomo.common.errors import PyomoException
from pyomo.common.timing import HierarchicalTimer
from pyomo.common.config import ConfigValue, NonNegativeInt
from pyomo.common.tee import TeeStream, capture_output
from pyomo.common.log import LogStream
from pyomo.core.kernel.objective import minimize, maximize
from pyomo.core.base import SymbolMap
from pyomo.core.base.var import _GeneralVarData
from pyomo.core.base.constraint import _GeneralConstraintData
from pyomo.core.base.sos import _SOSConstraintData
from pyomo.core.base.param import _ParamData
from pyomo.core.expr.numvalue import value, is_constant
from pyomo.repn import generate_standard_repn
from pyomo.core.expr.numeric_expr import NPV_MaxExpression, NPV_MinExpression
from pyomo.contrib.appsi.base import (
from pyomo.contrib.appsi.cmodel import cmodel, cmodel_available
from pyomo.common.dependencies import numpy as np
from pyomo.core.staleflag import StaleFlagManager
import sys
class _MutableConstraintBounds(object):

    def __init__(self, lower_expr, upper_expr, pyomo_con, con_map, highs):
        self.lower_expr = lower_expr
        self.upper_expr = upper_expr
        self.con = pyomo_con
        self.con_map = con_map
        self.highs = highs

    def update(self):
        row_ndx = self.con_map[self.con]
        lb = value(self.lower_expr)
        ub = value(self.upper_expr)
        self.highs.changeRowBounds(row_ndx, lb, ub)