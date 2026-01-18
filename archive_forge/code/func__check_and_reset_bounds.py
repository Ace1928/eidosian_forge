from collections import defaultdict
from pyomo.common.collections import ComponentMap, ComponentSet
from pyomo.contrib.fbbt.expression_bounds_walker import ExpressionBoundsVisitor
import pyomo.core.expr.numeric_expr as numeric_expr
from pyomo.core.expr.visitor import (
from pyomo.core.expr.numvalue import nonpyomo_leaf_types, value
from pyomo.core.expr.numvalue import is_fixed
import pyomo.contrib.fbbt.interval as interval
import math
from pyomo.core.base.block import Block
from pyomo.core.base.constraint import Constraint
from pyomo.core.base.var import Var
from pyomo.gdp import Disjunct
from pyomo.core.base.expression import _GeneralExpressionData, ScalarExpression
import logging
from pyomo.common.errors import InfeasibleConstraintException, PyomoException
from pyomo.common.config import (
from pyomo.common.numeric_types import native_types
from the constraint, we know that 1 <= x*y + z <= 1, so we may 
def _check_and_reset_bounds(var, lb, ub):
    """
    This function ensures that lb is not less than var.lb and that ub is not greater than var.ub.
    """
    orig_lb = var.lb
    orig_ub = var.ub
    if orig_lb is None:
        orig_lb = -interval.inf
    if orig_ub is None:
        orig_ub = interval.inf
    if lb < orig_lb:
        lb = orig_lb
    if ub > orig_ub:
        ub = orig_ub
    return (lb, ub)