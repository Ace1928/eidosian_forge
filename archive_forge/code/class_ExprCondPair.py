from sympy.core import S, Function, diff, Tuple, Dummy, Mul
from sympy.core.basic import Basic, as_Basic
from sympy.core.numbers import Rational, NumberSymbol, _illegal
from sympy.core.parameters import global_parameters
from sympy.core.relational import (Lt, Gt, Eq, Ne, Relational,
from sympy.core.sorting import ordered
from sympy.functions.elementary.miscellaneous import Max, Min
from sympy.logic.boolalg import (And, Boolean, distribute_and_over_or, Not,
from sympy.utilities.iterables import uniq, sift, common_prefix
from sympy.utilities.misc import filldedent, func_name
from itertools import product
class ExprCondPair(Tuple):
    """Represents an expression, condition pair."""

    def __new__(cls, expr, cond):
        expr = as_Basic(expr)
        if cond == True:
            return Tuple.__new__(cls, expr, true)
        elif cond == False:
            return Tuple.__new__(cls, expr, false)
        elif isinstance(cond, Basic) and cond.has(Piecewise):
            cond = piecewise_fold(cond)
            if isinstance(cond, Piecewise):
                cond = cond.rewrite(ITE)
        if not isinstance(cond, Boolean):
            raise TypeError(filldedent('\n                Second argument must be a Boolean,\n                not `%s`' % func_name(cond)))
        return Tuple.__new__(cls, expr, cond)

    @property
    def expr(self):
        """
        Returns the expression of this pair.
        """
        return self.args[0]

    @property
    def cond(self):
        """
        Returns the condition of this pair.
        """
        return self.args[1]

    @property
    def is_commutative(self):
        return self.expr.is_commutative

    def __iter__(self):
        yield self.expr
        yield self.cond

    def _eval_simplify(self, **kwargs):
        return self.func(*[a.simplify(**kwargs) for a in self.args])