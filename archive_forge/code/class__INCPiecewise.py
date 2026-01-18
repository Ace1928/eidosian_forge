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
class _INCPiecewise(object):
    """
    Called to generate Piecewise constraint using the INC formulation
    """

    def construct(self, pblock, x_var, y_var):
        x_pts = pblock._domain_pts
        y_pts = pblock._range_pts
        bound_type = pblock._bound_type
        if None in [x_pts, y_pts, bound_type]:
            raise RuntimeError('_INCPiecewise: construct() called during invalid state.')
        len_x_pts = len(x_pts)
        polytopes = range(1, len_x_pts)
        bin_y_index = range(1, len_x_pts - 1)
        pblock.INC_delta = Var(polytopes)
        delta = pblock.INC_delta
        delta[1].setub(1)
        delta[len_x_pts - 1].setlb(0)
        pblock.INC_bin_y = Var(bin_y_index, within=Binary)
        bin_y = pblock.INC_bin_y
        pblock.INC_constraint1 = Constraint(expr=x_var == x_pts[0] + sum((delta[p] * (x_pts[p] - x_pts[p - 1]) for p in polytopes)))
        LHS = y_var
        RHS = y_pts[0] + sum((delta[p] * (y_pts[p] - y_pts[p - 1]) for p in polytopes))
        expr = None
        if bound_type == Bound.Upper:
            expr = LHS <= RHS
        elif bound_type == Bound.Lower:
            expr = LHS >= RHS
        elif bound_type == Bound.Equal:
            expr = LHS == RHS
        else:
            raise ValueError('Invalid Bound for _INCPiecewise object')
        pblock.INC_constraint2 = Constraint(expr=expr)

        def con3_rule(model, p):
            if p != polytopes[-1]:
                return delta[p + 1] <= bin_y[p]
            else:
                return Constraint.Skip
        pblock.INC_constraint3 = Constraint(polytopes, rule=con3_rule)

        def con4_rule(model, p):
            if p != polytopes[-1]:
                return bin_y[p] <= delta[p]
            else:
                return Constraint.Skip
        pblock.INC_constraint4 = Constraint(polytopes, rule=con4_rule)