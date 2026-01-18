from pyomo.common.collections import ComponentMap, ComponentSet
import pyomo.core.expr as _expr
from pyomo.core.expr.visitor import ExpressionValueVisitor, nonpyomo_leaf_types
from pyomo.core.expr.numvalue import value, is_constant
from pyomo.core.expr import exp, log, sin, cos
import math
def _diff_DivisionExpression(node, val_dict, der_dict):
    """

    Parameters
    ----------
    node: pyomo.core.expr.numeric_expr.DivisionExpression
    val_dict: ComponentMap
    der_dict: ComponentMap
    """
    assert len(node.args) == 2
    num = node.args[0]
    den = node.args[1]
    der = der_dict[node]
    der_dict[num] += der * (1 / val_dict[den])
    der_dict[den] -= der * val_dict[num] / val_dict[den] ** 2