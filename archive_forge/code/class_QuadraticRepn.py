import copy
from pyomo.core.expr.numeric_expr import (
from pyomo.core.expr.relational_expr import (
from pyomo.core.base.expression import Expression
from . import linear
from .linear import _merge_dict, to_expression
class QuadraticRepn(object):
    __slots__ = ('multiplier', 'constant', 'linear', 'quadratic', 'nonlinear')

    def __init__(self):
        self.multiplier = 1
        self.constant = 0
        self.linear = {}
        self.quadratic = None
        self.nonlinear = None

    def __str__(self):
        return f'QuadraticRepn(mult={self.multiplier}, const={self.constant}, linear={self.linear}, quadratic={self.quadratic}, nonlinear={self.nonlinear})'

    def __repr__(self):
        return str(self)

    def walker_exitNode(self):
        if self.nonlinear is not None:
            return (_GENERAL, self)
        elif self.quadratic:
            return (_QUADRATIC, self)
        elif self.linear:
            return (_LINEAR, self)
        else:
            return (_CONSTANT, self.multiplier * self.constant)

    def duplicate(self):
        ans = self.__class__.__new__(self.__class__)
        ans.multiplier = self.multiplier
        ans.constant = self.constant
        ans.linear = dict(self.linear)
        if self.quadratic:
            ans.quadratic = dict(self.quadratic)
        else:
            ans.quadratic = None
        ans.nonlinear = self.nonlinear
        return ans

    def to_expression(self, visitor):
        var_map = visitor.var_map
        if self.nonlinear is not None:
            ans = self.nonlinear
        else:
            ans = 0
        if self.quadratic:
            with mutable_expression() as e:
                for (x1, x2), coef in self.quadratic.items():
                    if x1 == x2:
                        e += coef * var_map[x1] ** 2
                    else:
                        e += coef * (var_map[x1] * var_map[x2])
            ans += e
        if self.linear:
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
        so that we can directly use a QuadraticRepn() as a data object in
        the expression walker (thereby avoiding the function call for a
        custom callback)

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
        if other.quadratic:
            if not self.quadratic:
                self.quadratic = {}
            _merge_dict(self.quadratic, mult, other.quadratic)
        if other.nonlinear is not None:
            if mult != 1:
                nl = mult * other.nonlinear
            else:
                nl = other.nonlinear
            if self.nonlinear is None:
                self.nonlinear = nl
            else:
                self.nonlinear += nl