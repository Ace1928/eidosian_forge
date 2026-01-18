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
class QuadraticUDistribution(SingleContinuousDistribution):
    _argnames = ('a', 'b')

    @property
    def set(self):
        return Interval(self.a, self.b)

    @staticmethod
    def check(a, b):
        _value_check(b > a, 'Parameter b must be in range (%s, oo).' % a)

    def pdf(self, x):
        a, b = (self.a, self.b)
        alpha = 12 / (b - a) ** 3
        beta = (a + b) / 2
        return Piecewise((alpha * (x - beta) ** 2, And(a <= x, x <= b)), (S.Zero, True))

    def _moment_generating_function(self, t):
        a, b = (self.a, self.b)
        return -3 * (exp(a * t) * (4 + (a ** 2 + 2 * a * (-2 + b) + b ** 2) * t) - exp(b * t) * (4 + (-4 * b + (a + b) ** 2) * t)) / ((a - b) ** 3 * t ** 2)

    def _characteristic_function(self, t):
        a, b = (self.a, self.b)
        return -3 * I * (exp(I * a * t * exp(I * b * t)) * (4 * I - (-4 * b + (a + b) ** 2) * t)) / ((a - b) ** 3 * t ** 2)