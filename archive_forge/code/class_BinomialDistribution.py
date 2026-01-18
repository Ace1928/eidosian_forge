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
class BinomialDistribution(SingleFiniteDistribution):
    _argnames = ('n', 'p', 'succ', 'fail')

    @staticmethod
    def check(n, p, succ, fail):
        _value_check((n.is_integer, n.is_nonnegative), "'n' must be nonnegative integer.")
        _value_check((p <= 1, p >= 0), 'p should be in range [0, 1].')

    @property
    def high(self):
        return self.n

    @property
    def low(self):
        return S.Zero

    @property
    def is_symbolic(self):
        return not self.n.is_number

    @property
    def set(self):
        if self.is_symbolic:
            return Intersection(S.Naturals0, Interval(0, self.n))
        return set(self.dict.keys())

    def pmf(self, x):
        n, p = (self.n, self.p)
        x = sympify(x)
        if not (x.is_number or x.is_Symbol or is_random(x)):
            raise ValueError("'x' expected as an argument of type 'number', 'Symbol', or 'RandomSymbol' not %s" % type(x))
        cond = Ge(x, 0) & Le(x, n) & Contains(x, S.Integers)
        return Piecewise((binomial(n, x) * p ** x * (1 - p) ** (n - x), cond), (S.Zero, True))

    @property
    @cacheit
    def dict(self):
        if self.is_symbolic:
            return Density(self)
        return {k * self.succ + (self.n - k) * self.fail: self.pmf(k) for k in range(0, self.n + 1)}