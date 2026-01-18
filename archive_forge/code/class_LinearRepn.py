import logging
import sys
from operator import itemgetter
from itertools import filterfalse
from pyomo.common.deprecation import deprecation_warning
from pyomo.common.numeric_types import (
from pyomo.core.expr.numeric_expr import (
from pyomo.core.expr.relational_expr import (
from pyomo.core.expr.visitor import StreamBasedExpressionVisitor, _EvaluationVisitor
from pyomo.core.expr import is_fixed, value
from pyomo.core.base.expression import Expression
import pyomo.core.kernel as kernel
from pyomo.repn.util import (
class LinearRepn(object):
    __slots__ = ('multiplier', 'constant', 'linear', 'nonlinear')

    def __init__(self):
        self.multiplier = 1
        self.constant = 0
        self.linear = {}
        self.nonlinear = None

    def __str__(self):
        return f'LinearRepn(mult={self.multiplier}, const={self.constant}, linear={self.linear}, nonlinear={self.nonlinear})'

    def __repr__(self):
        return str(self)

    def walker_exitNode(self):
        if self.nonlinear is not None:
            return (_GENERAL, self)
        elif self.linear:
            return (_LINEAR, self)
        else:
            return (_CONSTANT, self.multiplier * self.constant)

    def duplicate(self):
        ans = self.__class__.__new__(self.__class__)
        ans.multiplier = self.multiplier
        ans.constant = self.constant
        ans.linear = dict(self.linear)
        ans.nonlinear = self.nonlinear
        return ans

    def to_expression(self, visitor):
        if self.nonlinear is not None:
            ans = self.nonlinear
        else:
            ans = 0
        if self.linear:
            var_map = visitor.var_map
            if len(self.linear) == 1:
                vid, coef = next(iter(self.linear.items()))
                if coef == 1:
                    ans += var_map[vid]
                elif coef:
                    ans += MonomialTermExpression((coef, var_map[vid]))
                else:
                    pass
            else:
                ans += LinearExpression([MonomialTermExpression((coef, var_map[vid])) for vid, coef in self.linear.items() if coef])
        if self.constant:
            ans += self.constant
        if self.multiplier != 1:
            ans *= self.multiplier
        return ans

    def append(self, other):
        """Append a child result from acceptChildResult

        Notes
        -----
        This method assumes that the operator was "+". It is implemented
        so that we can directly use a LinearRepn() as a `data` object in
        the expression walker (thereby allowing us to use the default
        implementation of acceptChildResult [which calls
        `data.append()`] and avoid the function call for a custom
        callback).

        """
        _type, other = other
        if _type is _CONSTANT:
            self.constant += other
            return
        mult = other.multiplier
        if not mult:
            return
        if other.constant:
            self.constant += mult * other.constant
        if other.linear:
            _merge_dict(self.linear, mult, other.linear)
        if other.nonlinear is not None:
            if mult != 1:
                nl = mult * other.nonlinear
            else:
                nl = other.nonlinear
            if self.nonlinear is None:
                self.nonlinear = nl
            else:
                self.nonlinear += nl