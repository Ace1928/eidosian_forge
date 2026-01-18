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
class IdealSolitonDistribution(SingleFiniteDistribution):
    _argnames = ('k',)

    @staticmethod
    def check(k):
        _value_check(k.is_integer and k.is_positive, "'k' must be a positive integer.")

    @property
    def low(self):
        return S.One

    @property
    def high(self):
        return self.k

    @property
    def set(self):
        return set(map(Integer, range(1, self.k + 1)))

    @property
    @cacheit
    def dict(self):
        if self.k.is_Symbol:
            return Density(self)
        d = {1: Rational(1, self.k)}
        d.update({i: Rational(1, i * (i - 1)) for i in range(2, self.k + 1)})
        return d

    def pmf(self, x):
        x = sympify(x)
        if not (x.is_number or x.is_Symbol or is_random(x)):
            raise ValueError("'x' expected as an argument of type 'number', 'Symbol', or 'RandomSymbol' not %s" % type(x))
        cond1 = Eq(x, 1) & x.is_integer
        cond2 = Ge(x, 1) & Le(x, self.k) & x.is_integer
        return Piecewise((1 / self.k, cond1), (1 / (x * (x - 1)), cond2), (S.Zero, True))