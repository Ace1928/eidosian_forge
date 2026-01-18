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
class StudentTDistribution(SingleContinuousDistribution):
    _argnames = ('nu',)
    set = Interval(-oo, oo)

    @staticmethod
    def check(nu):
        _value_check(nu > 0, 'Degrees of freedom nu must be positive.')

    def pdf(self, x):
        nu = self.nu
        return 1 / (sqrt(nu) * beta_fn(S.Half, nu / 2)) * (1 + x ** 2 / nu) ** (-(nu + 1) / 2)

    def _cdf(self, x):
        nu = self.nu
        return S.Half + x * gamma((nu + 1) / 2) * hyper((S.Half, (nu + 1) / 2), (Rational(3, 2),), -x ** 2 / nu) / (sqrt(pi * nu) * gamma(nu / 2))

    def _moment_generating_function(self, t):
        raise NotImplementedError('The moment generating function for the Student-T distribution is undefined.')