import operator
import sys
from pyomo.common import DeveloperError
from pyomo.common.collections import ComponentMap
from pyomo.common.dependencies import attempt_import
from pyomo.common.errors import NondifferentiableError
import pyomo.core.expr as EXPR
from pyomo.core.expr.numvalue import value, native_types
class Pyomo2SympyVisitor(EXPR.StreamBasedExpressionVisitor):

    def __init__(self, object_map):
        sympy.Add
        super(Pyomo2SympyVisitor, self).__init__()
        self.object_map = object_map

    def initializeWalker(self, expr):
        return self.beforeChild(None, expr, None)

    def exitNode(self, node, values):
        if node.__class__ is EXPR.UnaryFunctionExpression:
            return _functionMap[node._name](values[0])
        _op = _pyomo_operator_map.get(node.__class__, None)
        if _op is None:
            return node._apply_operation(values)
        else:
            return _op(*tuple(values))

    def beforeChild(self, node, child, child_idx):
        if type(child) in native_types:
            return (False, child)
        if child.is_potentially_variable():
            if child.is_expression_type():
                return (True, None)
            else:
                return (False, self.object_map.getSympySymbol(child))
        return (False, value(child))