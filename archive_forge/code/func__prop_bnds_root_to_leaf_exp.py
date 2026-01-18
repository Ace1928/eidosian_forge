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
def _prop_bnds_root_to_leaf_exp(node, bnds_dict, feasibility_tol):
    """

    Parameters
    ----------
    node: pyomo.core.expr.numeric_expr.ProductExpression
    bnds_dict: ComponentMap
    feasibility_tol: float
        If the bounds computed on the body of a constraint violate the bounds of the constraint by more than
        feasibility_tol, then the constraint is considered infeasible and an exception is raised. This tolerance
        is also used when performing certain interval arithmetic operations to ensure that none of the feasible
        region is removed due to floating point arithmetic and to prevent math domain errors (a larger value
        is more conservative).
    """
    assert len(node.args) == 1
    arg = node.args[0]
    lb0, ub0 = bnds_dict[node]
    lb1, ub1 = bnds_dict[arg]
    _lb1, _ub1 = interval.log(lb0, ub0)
    if _lb1 > lb1:
        lb1 = _lb1
    if _ub1 < ub1:
        ub1 = _ub1
    bnds_dict[arg] = (lb1, ub1)