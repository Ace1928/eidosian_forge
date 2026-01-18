from pyomo.contrib.appsi.base import (
from pyomo.core.expr.numeric_expr import (
from pyomo.common.errors import PyomoException
from pyomo.common.collections import ComponentMap
from pyomo.core.expr.numvalue import native_numeric_types
from typing import Dict, Optional, List
from pyomo.core.base.block import _BlockData
from pyomo.core.base.var import _GeneralVarData
from pyomo.core.base.param import _ParamData
from pyomo.core.base.constraint import _GeneralConstraintData
from pyomo.common.timing import HierarchicalTimer
from pyomo.core.base import SymbolMap, NumericLabeler, TextLabeler
from pyomo.common.dependencies import attempt_import
from pyomo.core.staleflag import StaleFlagManager
from pyomo.contrib.appsi.cmodel import cmodel, cmodel_available
import logging
import time
import sys
from pyomo.core.expr.visitor import ExpressionValueVisitor
class PyomoToWntrVisitor(ExpressionValueVisitor):

    def __init__(self, var_map, param_map):
        self.var_map = var_map
        self.param_map = param_map

    def visit(self, node, values):
        if node.__class__ in _handler_map:
            return _handler_map[node.__class__](node, values)
        else:
            raise NotImplementedError(f'Unrecognized expression type: {node.__class__}')

    def visiting_potential_leaf(self, node):
        if node.__class__ in native_numeric_types:
            return (True, node)
        if node.is_variable_type():
            return (True, self.var_map[id(node)])
        if node.is_parameter_type():
            return (True, self.param_map[id(node)])
        return (False, None)