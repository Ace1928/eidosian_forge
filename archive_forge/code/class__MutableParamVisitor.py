import inspect
import logging
import sys
from copy import deepcopy
from collections import deque
from pyomo.common.deprecation import deprecated, deprecation_warning
from pyomo.common.errors import DeveloperError, TemplateExpressionError
from pyomo.common.numeric_types import (
import pyomo.core.expr.expr_common as common
from pyomo.core.expr.symbol_map import SymbolMap
class _MutableParamVisitor(SimpleExpressionVisitor):

    def __init__(self):
        self.seen = set()

    def visit(self, node):
        if node.__class__ in nonpyomo_leaf_types:
            return
        if not node.is_variable_type() and node.is_fixed() and (not node.is_constant()):
            if id(node) in self.seen:
                return
            self.seen.add(id(node))
            return node