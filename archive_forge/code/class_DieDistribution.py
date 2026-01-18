from sympy.core.cache import cacheit
from sympy.core.function import Lambda
from sympy.core.numbers import (Integer, Rational)
from sympy.core.relational import (Eq, Ge, Gt, Le, Lt)
from sympy.core.singleton import S
from sympy.core.symbol import (Dummy, Symbol)
from sympy.core.sympify import sympify
from sympy.functions.combinatorial.factorials import binomial
from sympy.functions.elementary.exponential import log
from sympy.functions.elementary.piecewise import Piecewise
from sympy.logic.boolalg import Or
from sympy.sets.contains import Contains
from sympy.sets.fancysets import Range
from sympy.sets.sets import (Intersection, Interval)
from sympy.functions.special.beta_functions import beta as beta_fn
from sympy.stats.frv import (SingleFiniteDistribution,
from sympy.stats.rv import _value_check, Density, is_random
from sympy.utilities.iterables import multiset
from sympy.utilities.misc import filldedent
class DieDistribution(SingleFiniteDistribution):
    _argnames = ('sides',)

    @staticmethod
    def check(sides):
        _value_check((sides.is_positive, sides.is_integer), 'number of sides must be a positive integer.')

    @property
    def is_symbolic(self):
        return not self.sides.is_number

    @property
    def high(self):
        return self.sides

    @property
    def low(self):
        return S.One

    @property
    def set(self):
        if self.is_symbolic:
            return Intersection(S.Naturals0, Interval(0, self.sides))
        return set(map(Integer, range(1, self.sides + 1)))

    def pmf(self, x):
        x = sympify(x)
        if not (x.is_number or x.is_Symbol or is_random(x)):
            raise ValueError("'x' expected as an argument of type 'number', 'Symbol', or 'RandomSymbol' not %s" % type(x))
        cond = Ge(x, 1) & Le(x, self.sides) & Contains(x, S.Integers)
        return Piecewise((S.One / self.sides, cond), (S.Zero, True))