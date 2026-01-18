from pyomo.common.collections import ComponentMap, ComponentSet
import pyomo.core.expr as _expr
from pyomo.core.expr.visitor import ExpressionValueVisitor, nonpyomo_leaf_types
from pyomo.core.expr.numvalue import value, is_constant
from pyomo.core.expr import exp, log, sin, cos
import math
def _diff_ProductExpression(node, val_dict, der_dict):
    """

    Parameters
    ----------
    node: pyomo.core.expr.numeric_expr.ProductExpression
    val_dict: ComponentMap
    der_dict: ComponentMap
    """
    assert len(node.args) == 2
    arg1, arg2 = node.args
    der = der_dict[node]
    der_dict[arg1] += der * val_dict[arg2]
    der_dict[arg2] += der * val_dict[arg1]