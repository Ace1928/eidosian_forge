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
def Arcsin(name, a=0, b=1):
    """
    Create a Continuous Random Variable with an arcsin distribution.

    The density of the arcsin distribution is given by

    .. math::
        f(x) := \\frac{1}{\\pi\\sqrt{(x-a)(b-x)}}

    with :math:`x \\in (a,b)`. It must hold that :math:`-\\infty < a < b < \\infty`.

    Parameters
    ==========

    a : Real number, the left interval boundary
    b : Real number, the right interval boundary

    Returns
    =======

    RandomSymbol

    Examples
    ========

    >>> from sympy.stats import Arcsin, density, cdf
    >>> from sympy import Symbol

    >>> a = Symbol("a", real=True)
    >>> b = Symbol("b", real=True)
    >>> z = Symbol("z")

    >>> X = Arcsin("x", a, b)

    >>> density(X)(z)
    1/(pi*sqrt((-a + z)*(b - z)))

    >>> cdf(X)(z)
    Piecewise((0, a > z),
            (2*asin(sqrt((-a + z)/(-a + b)))/pi, b >= z),
            (1, True))


    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Arcsine_distribution

    """
    return rv(name, ArcsinDistribution, (a, b))