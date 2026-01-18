from pyomo.common.collections import ComponentMap, ComponentSet
import pyomo.core.expr as _expr
from pyomo.core.expr.visitor import ExpressionValueVisitor, nonpyomo_leaf_types
from pyomo.core.expr.numvalue import value, is_constant
from pyomo.core.expr import exp, log, sin, cos
import math
def _diff_atan(node, val_dict, der_dict):
    """

    Parameters
    ----------
    node: pyomo.core.expr.numeric_expr.UnaryFunctionExpression
    val_dict: ComponentMap
    der_dict: ComponentMap
    """
    assert len(node.args) == 1
    arg = node.args[0]
    der = der_dict[node]
    der_dict[arg] += der / (1 + val_dict[arg] ** 2)