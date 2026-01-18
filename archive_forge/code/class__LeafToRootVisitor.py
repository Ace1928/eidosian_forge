from pyomo.common.collections import ComponentMap, ComponentSet
import pyomo.core.expr as _expr
from pyomo.core.expr.visitor import ExpressionValueVisitor, nonpyomo_leaf_types
from pyomo.core.expr.numvalue import value, is_constant
from pyomo.core.expr import exp, log, sin, cos
import math
class _LeafToRootVisitor(ExpressionValueVisitor):

    def __init__(self, val_dict, der_dict, expr_list, numeric=True):
        """
        Parameters
        ----------
        val_dict: ComponentMap
        der_dict: ComponentMap
        """
        self.val_dict = val_dict
        self.der_dict = der_dict
        self.expr_list = expr_list
        assert len(self.expr_list) == 0
        assert len(self.val_dict) == 0
        assert len(self.der_dict) == 0
        if numeric:
            self.value_func = value
            self.operation_func = _numeric_apply_operation
        else:
            self.value_func = _symbolic_value
            self.operation_func = _symbolic_apply_operation

    def visit(self, node, values):
        self.val_dict[node] = self.operation_func(node, values)
        self.der_dict[node] = 0
        self.expr_list.append(node)
        return self.val_dict[node]

    def visiting_potential_leaf(self, node):
        if node in self.val_dict:
            return (True, self.val_dict[node])
        if node.__class__ in nonpyomo_leaf_types:
            self.val_dict[node] = node
            self.der_dict[node] = 0
            return (True, node)
        if not node.is_expression_type():
            val = self.value_func(node)
            self.val_dict[node] = val
            self.der_dict[node] = 0
            return (True, val)
        return (False, None)