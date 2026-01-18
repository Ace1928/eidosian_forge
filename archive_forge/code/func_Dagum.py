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
def Dagum(name, p, a, b):
    """
    Create a continuous random variable with a Dagum distribution.

    Explanation
    ===========

    The density of the Dagum distribution is given by

    .. math::
        f(x) := \\frac{a p}{x} \\left( \\frac{\\left(\\tfrac{x}{b}\\right)^{a p}}
                {\\left(\\left(\\tfrac{x}{b}\\right)^a + 1 \\right)^{p+1}} \\right)

    with :math:`x > 0`.

    Parameters
    ==========

    p : Real number
        `p > 0`, a shape.
    a : Real number
        `a > 0`, a shape.
    b : Real number
        `b > 0`, a scale.

    Returns
    =======

    RandomSymbol

    Examples
    ========

    >>> from sympy.stats import Dagum, density, cdf
    >>> from sympy import Symbol

    >>> p = Symbol("p", positive=True)
    >>> a = Symbol("a", positive=True)
    >>> b = Symbol("b", positive=True)
    >>> z = Symbol("z")

    >>> X = Dagum("x", p, a, b)

    >>> density(X)(z)
    a*p*(z/b)**(a*p)*((z/b)**a + 1)**(-p - 1)/z

    >>> cdf(X)(z)
    Piecewise(((1 + (z/b)**(-a))**(-p), z >= 0), (0, True))


    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Dagum_distribution

    """
    return rv(name, DagumDistribution, (p, a, b))