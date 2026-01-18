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
class FiniteDistributionHandmade(SingleFiniteDistribution):

    @property
    def dict(self):
        return self.args[0]

    def pmf(self, x):
        x = Symbol('x')
        return Lambda(x, Piecewise(*[(v, Eq(k, x)) for k, v in self.dict.items()] + [(S.Zero, True)]))

    @property
    def set(self):
        return set(self.dict.keys())

    @staticmethod
    def check(density):
        for p in density.values():
            _value_check((p >= 0, p <= 1), 'Probability at a point must be between 0 and 1.')
        val = sum(density.values())
        _value_check(Eq(val, 1) != S.false, 'Total Probability must be 1.')