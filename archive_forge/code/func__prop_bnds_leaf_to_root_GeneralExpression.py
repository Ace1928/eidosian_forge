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
def _prop_bnds_leaf_to_root_GeneralExpression(visitor, node, expr):
    """
    Propagate bounds from children to parent

    Parameters
    ----------
    visitor: _FBBTVisitorLeafToRoot
    node: pyomo.core.base.expression._GeneralExpressionData
    expr: GeneralExpression arg
    """
    bnds_dict = visitor.bnds_dict
    if node in bnds_dict:
        return
    if expr.__class__ in native_types:
        expr_lb = expr_ub = expr
    else:
        expr_lb, expr_ub = bnds_dict[expr]
    bnds_dict[node] = (expr_lb, expr_ub)