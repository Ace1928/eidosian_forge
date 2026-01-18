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
class _EvaluateConstantExpressionVisitor(ExpressionValueVisitor):

    def visit(self, node, values):
        """Visit nodes that have been expanded"""
        return node._apply_operation(values)

    def visiting_potential_leaf(self, node):
        """
        Visiting a potential leaf.

        Return True if the node is not expanded.
        """
        if node.__class__ in nonpyomo_leaf_types:
            return (True, node)
        if node.is_expression_type():
            return (False, None)
        if node.is_numeric_type():
            try:
                val = value(node)
            except TemplateExpressionError:
                raise
            except:
                if not node.is_fixed():
                    raise NonConstantExpressionError()
                if not node.is_constant():
                    raise FixedExpressionError()
                raise
            if not node.is_fixed():
                raise NonConstantExpressionError()
            if not node.is_constant():
                raise FixedExpressionError()
            return (True, val)
        return (True, node)