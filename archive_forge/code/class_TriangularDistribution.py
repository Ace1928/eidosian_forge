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
class TriangularDistribution(SingleContinuousDistribution):
    _argnames = ('a', 'b', 'c')

    @property
    def set(self):
        return Interval(self.a, self.b)

    @staticmethod
    def check(a, b, c):
        _value_check(b > a, 'Parameter b > %s. b = %s' % (a, b))
        _value_check((a <= c, c <= b), 'Parameter c must be in range [%s, %s]. c = %s' % (a, b, c))

    def pdf(self, x):
        a, b, c = (self.a, self.b, self.c)
        return Piecewise((2 * (x - a) / ((b - a) * (c - a)), And(a <= x, x < c)), (2 / (b - a), Eq(x, c)), (2 * (b - x) / ((b - a) * (b - c)), And(c < x, x <= b)), (S.Zero, True))

    def _characteristic_function(self, t):
        a, b, c = (self.a, self.b, self.c)
        return -2 * ((b - c) * exp(I * a * t) - (b - a) * exp(I * c * t) + (c - a) * exp(I * b * t)) / ((b - a) * (c - a) * (b - c) * t ** 2)

    def _moment_generating_function(self, t):
        a, b, c = (self.a, self.b, self.c)
        return 2 * ((b - c) * exp(a * t) - (b - a) * exp(c * t) + (c - a) * exp(b * t)) / ((b - a) * (c - a) * (b - c) * t ** 2)