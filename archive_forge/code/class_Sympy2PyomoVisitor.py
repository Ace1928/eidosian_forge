import operator
import sys
from pyomo.common import DeveloperError
from pyomo.common.collections import ComponentMap
from pyomo.common.dependencies import attempt_import
from pyomo.common.errors import NondifferentiableError
import pyomo.core.expr as EXPR
from pyomo.core.expr.numvalue import value, native_types
class Sympy2PyomoVisitor(EXPR.StreamBasedExpressionVisitor):

    def __init__(self, object_map):
        sympy.Add
        super(Sympy2PyomoVisitor, self).__init__()
        self.object_map = object_map

    def initializeWalker(self, expr):
        return self.beforeChild(None, expr, None)

    def enterNode(self, node):
        return (node.args, [])

    def exitNode(self, node, values):
        """Visit nodes that have been expanded"""
        _op = _operatorMap.get(node.func, None)
        if _op is None:
            raise DeveloperError(f'sympy expression type {node.func} not found in the operator map')
        return _op(tuple(values))

    def beforeChild(self, node, child, child_idx):
        if not child.args:
            item = self.object_map.getPyomoSymbol(child, None)
            if item is None:
                item = float(child.evalf())
            return (False, item)
        return (True, None)