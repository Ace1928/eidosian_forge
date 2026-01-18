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
class _DCCPiecewise(object):
    """
    Called to generate Piecewise constraint using the DCC formulation
    """

    def construct(self, pblock, x_var, y_var):
        x_pts = pblock._domain_pts
        y_pts = pblock._range_pts
        bound_type = pblock._bound_type
        if None in [x_pts, y_pts, bound_type]:
            raise RuntimeError('_DCCPiecewise: construct() called during invalid state.')
        len_x_pts = len(x_pts)
        polytopes = range(1, len_x_pts)
        vertices = range(1, len_x_pts + 1)

        def polytope_verts(p):
            return range(p, p + 2)
        pblock.DCC_lambda = Var(polytopes, vertices, within=PositiveReals)
        lmda = pblock.DCC_lambda
        pblock.DCC_bin_y = Var(polytopes, within=Binary)
        bin_y = pblock.DCC_bin_y
        pblock.DCC_constraint1 = Constraint(expr=x_var == sum((lmda[p, v] * x_pts[v - 1] for p in polytopes for v in polytope_verts(p))))
        LHS = y_var
        RHS = sum((lmda[p, v] * y_pts[v - 1] for p in polytopes for v in polytope_verts(p)))
        expr = None
        if bound_type == Bound.Upper:
            expr = LHS <= RHS
        elif bound_type == Bound.Lower:
            expr = LHS >= RHS
        elif bound_type == Bound.Equal:
            expr = LHS == RHS
        else:
            raise ValueError('Invalid Bound for _DCCPiecewise object')
        pblock.DCC_constraint2 = Constraint(expr=expr)

        def con3_rule(model, p):
            return bin_y[p] == sum((lmda[p, v] for v in polytope_verts(p)))
        pblock.DCC_constraint3 = Constraint(polytopes, rule=con3_rule)
        pblock.DCC_constraint4 = Constraint(expr=sum((bin_y[p] for p in polytopes)) == 1)