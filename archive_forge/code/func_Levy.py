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
def Levy(name, mu, c):
    """
    Create a continuous random variable with a Levy distribution.

    The density of the Levy distribution is given by

    .. math::
        f(x) := \\sqrt(\\frac{c}{2 \\pi}) \\frac{\\exp -\\frac{c}{2 (x - \\mu)}}{(x - \\mu)^{3/2}}

    Parameters
    ==========

    mu : Real number
        The location parameter.
    c : Real number, `c > 0`
        A scale parameter.

    Returns
    =======

    RandomSymbol

    Examples
    ========

    >>> from sympy.stats import Levy, density, cdf
    >>> from sympy import Symbol

    >>> mu = Symbol("mu", real=True)
    >>> c = Symbol("c", positive=True)
    >>> z = Symbol("z")

    >>> X = Levy("x", mu, c)

    >>> density(X)(z)
    sqrt(2)*sqrt(c)*exp(-c/(-2*mu + 2*z))/(2*sqrt(pi)*(-mu + z)**(3/2))

    >>> cdf(X)(z)
    erfc(sqrt(c)*sqrt(1/(-2*mu + 2*z)))

    References
    ==========
    .. [1] https://en.wikipedia.org/wiki/L%C3%A9vy_distribution
    .. [2] https://mathworld.wolfram.com/LevyDistribution.html
    """
    return rv(name, LevyDistribution, (mu, c))