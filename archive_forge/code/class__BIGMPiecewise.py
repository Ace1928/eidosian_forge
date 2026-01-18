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
class _BIGMPiecewise(object):
    """
    Called to generate Piecewise constraint using the BIGM formulation
    """

    def __init__(self, binary=True):
        self.binary = binary
        if not self.binary in [True, False]:
            raise ValueError('_BIGMPiecewise must be initialized with the binary flag set to True or False (choose one).')

    def construct(self, pblock, x_var, y_var):
        tag = ''
        x_pts = pblock._domain_pts
        y_pts = pblock._range_pts
        bound_type = pblock._bound_type
        if None in [x_pts, y_pts, bound_type]:
            raise RuntimeError('_BIGMPiecewise: construct() called during invalid state.')
        len_x_pts = len(x_pts)
        if self.binary is True:
            tag += 'bin'
        else:
            tag += 'sos1'
        OPT_M = {}
        OPT_M['UB'] = {}
        OPT_M['LB'] = {}
        if bound_type in [Bound.Upper, Bound.Equal]:
            OPT_M['UB'] = self._find_M(x_pts, y_pts, Bound.Upper)
        if bound_type in [Bound.Lower, Bound.Equal]:
            OPT_M['LB'] = self._find_M(x_pts, y_pts, Bound.Lower)
        all_keys = set(OPT_M['UB'].keys()).union(OPT_M['LB'].keys())
        full_indices = []
        full_indices.extend(range(1, len_x_pts))
        bigm_y_index = None
        bigm_y = None
        if len(all_keys) > 0:
            bigm_y_index = all_keys

            def y_domain():
                if self.binary is True:
                    return Binary
                else:
                    return NonNegativeReals
            setattr(pblock, tag + '_y', Var(bigm_y_index, within=y_domain()))
            bigm_y = getattr(pblock, tag + '_y')

        def con1_rule(model, i):
            if bound_type in [Bound.Upper, Bound.Equal]:
                rhs = 1.0
                if i not in OPT_M['UB']:
                    rhs *= 0.0
                else:
                    rhs *= OPT_M['UB'][i] * (1 - bigm_y[i])
                return y_var - y_pts[i - 1] - (y_pts[i] - y_pts[i - 1]) / (x_pts[i] - x_pts[i - 1]) * (x_var - x_pts[i - 1]) <= rhs
            elif bound_type == Bound.Lower:
                rhs = 1.0
                if i not in OPT_M['LB']:
                    rhs *= 0.0
                else:
                    rhs *= OPT_M['LB'][i] * (1 - bigm_y[i])
                return y_var - y_pts[i - 1] - (y_pts[i] - y_pts[i - 1]) / (x_pts[i] - x_pts[i - 1]) * (x_var - x_pts[i - 1]) >= rhs

        def con2_rule(model):
            expr = [bigm_y[i] for i in range(1, len_x_pts) if i in all_keys]
            if len(expr) > 0:
                return sum(expr) == 1
            else:
                return Constraint.Skip

        def conAFF_rule(model, i):
            rhs = 1.0
            if i not in OPT_M['LB']:
                rhs *= 0.0
            else:
                rhs *= OPT_M['LB'][i] * (1 - bigm_y[i])
            return y_var - y_pts[i - 1] - (y_pts[i] - y_pts[i - 1]) / (x_pts[i] - x_pts[i - 1]) * (x_var - x_pts[i - 1]) >= rhs
        pblock.BIGM_constraint1 = Constraint(full_indices, rule=con1_rule)
        if len(all_keys) > 0:
            pblock.BIGM_constraint2 = Constraint(rule=con2_rule)
        if bound_type == Bound.Equal:
            pblock.BIGM_constraint3 = Constraint(full_indices, rule=conAFF_rule)
        if len(all_keys) > 0:
            if self.binary is False:
                pblock.BIGM_constraint4 = SOSConstraint(var=bigm_y, sos=1)
        if not x_var.lb is None and x_var.lb < x_pts[0]:
            pblock.bigm_domain_constraint_lower = Constraint(expr=x_pts[0] <= x_var)
        if not x_var.ub is None and x_var.ub > x_pts[-1]:
            pblock.bigm_domain_constraint_upper = Constraint(expr=x_var <= x_pts[-1])

    def _M_func(self, a, Fa, b, Fb, c, Fc):
        return Fa - Fb - (a - b) * ((Fc - Fb) / (c - b))

    def _find_M(self, x_pts, y_pts, bound_type):
        len_x_pts = len(x_pts)
        _self_M_func = self._M_func
        M_final = {}
        for j in range(1, len_x_pts):
            index = j
            if bound_type == Bound.Lower:
                M_final[index] = min([0.0, min([_self_M_func(x_pts[k], y_pts[k], x_pts[j - 1], y_pts[j - 1], x_pts[j], y_pts[j]) for k in range(len_x_pts)])])
            elif bound_type == Bound.Upper:
                M_final[index] = max([0.0, max([_self_M_func(x_pts[k], y_pts[k], x_pts[j - 1], y_pts[j - 1], x_pts[j], y_pts[j]) for k in range(len_x_pts)])])
            else:
                raise ValueError('Invalid Bound passed to _find_M function')
            if M_final[index] == 0.0:
                del M_final[index]
        return M_final