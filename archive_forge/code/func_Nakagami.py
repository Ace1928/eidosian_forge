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
def Nakagami(name, mu, omega):
    """
    Create a continuous random variable with a Nakagami distribution.

    Explanation
    ===========

    The density of the Nakagami distribution is given by

    .. math::
        f(x) := \\frac{2\\mu^\\mu}{\\Gamma(\\mu)\\omega^\\mu} x^{2\\mu-1}
                \\exp\\left(-\\frac{\\mu}{\\omega}x^2 \\right)

    with :math:`x > 0`.

    Parameters
    ==========

    mu : Real number, `\\mu \\geq \\frac{1}{2}`, a shape
    omega : Real number, `\\omega > 0`, the spread

    Returns
    =======

    RandomSymbol

    Examples
    ========

    >>> from sympy.stats import Nakagami, density, E, variance, cdf
    >>> from sympy import Symbol, simplify, pprint

    >>> mu = Symbol("mu", positive=True)
    >>> omega = Symbol("omega", positive=True)
    >>> z = Symbol("z")

    >>> X = Nakagami("x", mu, omega)

    >>> D = density(X)(z)
    >>> pprint(D, use_unicode=False)
                                    2
                               -mu*z
                               -------
        mu      -mu  2*mu - 1  omega
    2*mu  *omega   *z        *e
    ----------------------------------
                Gamma(mu)

    >>> simplify(E(X))
    sqrt(mu)*sqrt(omega)*gamma(mu + 1/2)/gamma(mu + 1)

    >>> V = simplify(variance(X))
    >>> pprint(V, use_unicode=False)
                        2
             omega*Gamma (mu + 1/2)
    omega - -----------------------
            Gamma(mu)*Gamma(mu + 1)

    >>> cdf(X)(z)
    Piecewise((lowergamma(mu, mu*z**2/omega)/gamma(mu), z > 0),
            (0, True))


    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Nakagami_distribution

    """
    return rv(name, NakagamiDistribution, (mu, omega))