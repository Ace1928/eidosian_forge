import logging
import math
import itertools
import operator
import types
import enum
from pyomo.common.log import is_debug_set
from pyomo.common.deprecation import deprecation_warning
from pyomo.common.numeric_types import value
from pyomo.common.timing import ConstructionTimer
from pyomo.core.base.block import Block, _BlockData
from pyomo.core.base.component import ModelComponentFactory
from pyomo.core.base.constraint import Constraint, ConstraintList
from pyomo.core.base.sos import SOSConstraint
from pyomo.core.base.var import Var, _VarData, IndexedVar
from pyomo.core.base.set_types import PositiveReals, NonNegativeReals, Binary
from pyomo.core.base.util import flatten_tuple
class _SimpleSinglePiecewise(object):
    """
    Called when the piecewise points list has only two points
    """

    def construct(self, pblock, x_var, y_var):
        x_pts = pblock._domain_pts
        y_pts = pblock._range_pts
        bound_type = pblock._bound_type
        if None in [x_pts, y_pts, bound_type]:
            raise RuntimeError('_SimpleSinglePiecewise: construct() called duringinvalid state.')
        LHS = y_var
        F_AT_XO = y_pts[0]
        dF_AT_XO = (y_pts[1] - y_pts[0]) / (x_pts[1] - x_pts[0])
        X_MINUS_XO = x_var - x_pts[0]
        if bound_type == Bound.Upper:
            expr = LHS <= F_AT_XO + dF_AT_XO * X_MINUS_XO
        elif bound_type == Bound.Lower:
            expr = LHS >= F_AT_XO + dF_AT_XO * X_MINUS_XO
        elif bound_type == Bound.Equal:
            expr = LHS == F_AT_XO + dF_AT_XO * X_MINUS_XO
        else:
            raise ValueError('Invalid Bound for _SimpleSinglePiecewise object')
        pblock.single_line_constraint = Constraint(expr=expr)
        if not x_var.lb is None and x_var.lb < x_pts[0]:
            pblock.simplified_piecewise_domain_constraint_lower = Constraint(expr=x_pts[0] <= x_var)
        if not x_var.ub is None and x_var.ub > x_pts[1]:
            pblock.simplified_piecewise_domain_constraint_upper = Constraint(expr=x_var <= x_pts[-1])