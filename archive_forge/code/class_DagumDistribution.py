from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.trigonometric import (atan, cos, sin, tan)
from sympy.functions.special.bessel import (besseli, besselj, besselk)
from sympy.functions.special.beta_functions import beta as beta_fn
from sympy.concrete.summations import Sum
from sympy.core.basic import Basic
from sympy.core.function import Lambda
from sympy.core.numbers import (I, Rational, pi)
from sympy.core.relational import (Eq, Ne)
from sympy.core.singleton import S
from sympy.core.symbol import Dummy
from sympy.core.sympify import sympify
from sympy.functions.combinatorial.factorials import (binomial, factorial)
from sympy.functions.elementary.complexes import (Abs, sign)
from sympy.functions.elementary.exponential import log
from sympy.functions.elementary.hyperbolic import sinh
from sympy.functions.elementary.integers import floor
from sympy.functions.elementary.miscellaneous import sqrt, Max, Min
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.elementary.trigonometric import asin
from sympy.functions.special.error_functions import (erf, erfc, erfi, erfinv, expint)
from sympy.functions.special.gamma_functions import (gamma, lowergamma, uppergamma)
from sympy.functions.special.hyper import hyper
from sympy.integrals.integrals import integrate
from sympy.logic.boolalg import And
from sympy.sets.sets import Interval
from sympy.matrices import MatrixBase
from sympy.stats.crv import SingleContinuousPSpace, SingleContinuousDistribution
from sympy.stats.rv import _value_check, is_random
class DagumDistribution(SingleContinuousDistribution):
    _argnames = ('p', 'a', 'b')
    set = Interval(0, oo)

    @staticmethod
    def check(p, a, b):
        _value_check(p > 0, 'Shape parameter p must be positive.')
        _value_check(a > 0, 'Shape parameter a must be positive.')
        _value_check(b > 0, 'Scale parameter b must be positive.')

    def pdf(self, x):
        p, a, b = (self.p, self.a, self.b)
        return a * p / x * ((x / b) ** (a * p) / ((x / b) ** a + 1) ** (p + 1))

    def _cdf(self, x):
        p, a, b = (self.p, self.a, self.b)
        return Piecewise(((S.One + (S(x) / b) ** (-a)) ** (-p), x >= 0), (S.Zero, True))