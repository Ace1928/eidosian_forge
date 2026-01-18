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
def Gumbel(name, beta, mu, minimum=False):
    """
    Create a Continuous Random Variable with Gumbel distribution.

    Explanation
    ===========

    The density of the Gumbel distribution is given by

    For Maximum

    .. math::
        f(x) := \\dfrac{1}{\\beta} \\exp \\left( -\\dfrac{x-\\mu}{\\beta}
                - \\exp \\left( -\\dfrac{x - \\mu}{\\beta} \\right) \\right)

    with :math:`x \\in [ - \\infty, \\infty ]`.

    For Minimum

    .. math::
        f(x) := \\frac{e^{- e^{\\frac{- \\mu + x}{\\beta}} + \\frac{- \\mu + x}{\\beta}}}{\\beta}

    with :math:`x \\in [ - \\infty, \\infty ]`.

    Parameters
    ==========

    mu : Real number, `\\mu`, a location
    beta : Real number, `\\beta > 0`, a scale
    minimum : Boolean, by default ``False``, set to ``True`` for enabling minimum distribution

    Returns
    =======

    RandomSymbol

    Examples
    ========

    >>> from sympy.stats import Gumbel, density, cdf
    >>> from sympy import Symbol
    >>> x = Symbol("x")
    >>> mu = Symbol("mu")
    >>> beta = Symbol("beta", positive=True)
    >>> X = Gumbel("x", beta, mu)
    >>> density(X)(x)
    exp(-exp(-(-mu + x)/beta) - (-mu + x)/beta)/beta
    >>> cdf(X)(x)
    exp(-exp(-(-mu + x)/beta))

    References
    ==========

    .. [1] https://mathworld.wolfram.com/GumbelDistribution.html
    .. [2] https://en.wikipedia.org/wiki/Gumbel_distribution
    .. [3] https://web.archive.org/web/20200628222206/http://www.mathwave.com/help/easyfit/html/analyses/distributions/gumbel_max.html
    .. [4] https://web.archive.org/web/20200628222212/http://www.mathwave.com/help/easyfit/html/analyses/distributions/gumbel_min.html

    """
    return rv(name, GumbelDistribution, (beta, mu, minimum))