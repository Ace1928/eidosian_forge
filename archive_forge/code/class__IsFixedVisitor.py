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
class _IsFixedVisitor(ExpressionValueVisitor):
    """
    NOTE: This doesn't check if combiner logic is
    all or any and short-circuit the test.  It's
    not clear that that is an important optimization.
    """

    def visit(self, node, values):
        """Visit nodes that have been expanded"""
        return node._is_fixed(values)

    def visiting_potential_leaf(self, node):
        """
        Visiting a potential leaf.

        Return True if the node is not expanded.
        """
        if node.__class__ in nonpyomo_leaf_types:
            return (True, True)
        elif node.is_expression_type():
            return (False, None)
        elif node.is_numeric_type():
            return (True, node.is_fixed())
        return (True, node)