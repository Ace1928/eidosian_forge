from pyomo.common.collections import ComponentMap, ComponentSet
import pyomo.core.expr as _expr
from pyomo.core.expr.visitor import ExpressionValueVisitor, nonpyomo_leaf_types
from pyomo.core.expr.numvalue import value, is_constant
from pyomo.core.expr import exp, log, sin, cos
import math
def _diff_GeneralExpression(node, val_dict, der_dict):
    """
    Reverse automatic differentiation for named expressions.

    Parameters
    ----------
    node: The named expression
    val_dict: ComponentMap
    der_dict: ComponentMap
    """
    der_dict[node.arg(0)] += der_dict[node]