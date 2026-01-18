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
def LogCauchy(name, mu, sigma):
    """
    Create a continuous random variable with a Log-Cauchy distribution.
    The density of the Log-Cauchy distribution is given by

    .. math::
        f(x) := \\frac{1}{\\pi x} \\frac{\\sigma}{(log(x)-\\mu^2) + \\sigma^2}

    Parameters
    ==========

    mu : Real number, the location

    sigma : Real number, `\\sigma > 0`, a scale

    Returns
    =======

    RandomSymbol

    Examples
    ========

    >>> from sympy.stats import LogCauchy, density, cdf
    >>> from sympy import Symbol, S

    >>> mu = 2
    >>> sigma = S.One / 5
    >>> z = Symbol("z")

    >>> X = LogCauchy("x", mu, sigma)

    >>> density(X)(z)
    1/(5*pi*z*((log(z) - 2)**2 + 1/25))

    >>> cdf(X)(z)
    atan(5*log(z) - 10)/pi + 1/2

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Log-Cauchy_distribution
    """
    return rv(name, LogCauchyDistribution, (mu, sigma))