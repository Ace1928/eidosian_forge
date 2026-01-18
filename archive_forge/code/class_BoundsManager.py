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
class BoundsManager(object):

    def __init__(self, comp):
        self._vars = ComponentSet()
        self._saved_bounds = list()
        if comp.ctype == Constraint:
            if comp.is_indexed():
                for c in comp.values():
                    self._vars.update(identify_variables(c.body))
            else:
                self._vars.update(identify_variables(comp.body))
        else:
            for c in comp.component_data_objects(Constraint, descend_into=True, active=True, sort=True):
                self._vars.update(identify_variables(c.body))

    def save_bounds(self):
        bnds = ComponentMap()
        for v in self._vars:
            bnds[v] = (v.lb, v.ub)
        self._saved_bounds.append(bnds)

    def pop_bounds(self, ndx=-1):
        bnds = self._saved_bounds.pop(ndx)
        for v, _bnds in bnds.items():
            lb, ub = _bnds
            v.setlb(lb)
            v.setub(ub)

    def load_bounds(self, bnds, save_current_bounds=True):
        if save_current_bounds:
            self.save_bounds()
        for v, _bnds in bnds.items():
            if v in self._vars:
                lb, ub = _bnds
                v.setlb(lb)
                v.setub(ub)