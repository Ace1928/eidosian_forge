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
def RaisedCosine(name, mu, s):
    """
    Create a Continuous Random Variable with a raised cosine distribution.

    Explanation
    ===========

    The density of the raised cosine distribution is given by

    .. math::
        f(x) := \\frac{1}{2s}\\left(1+\\cos\\left(\\frac{x-\\mu}{s}\\pi\\right)\\right)

    with :math:`x \\in [\\mu-s,\\mu+s]`.

    Parameters
    ==========

    mu : Real number
    s : Real number, `s > 0`

    Returns
    =======

    RandomSymbol

    Examples
    ========

    >>> from sympy.stats import RaisedCosine, density
    >>> from sympy import Symbol, pprint

    >>> mu = Symbol("mu", real=True)
    >>> s = Symbol("s", positive=True)
    >>> z = Symbol("z")

    >>> X = RaisedCosine("x", mu, s)

    >>> D = density(X)(z)
    >>> pprint(D, use_unicode=False)
    /   /pi*(-mu + z)\\
    |cos|------------| + 1
    |   \\     s      /
    <---------------------  for And(z >= mu - s, z <= mu + s)
    |         2*s
    |
    \\          0                        otherwise

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Raised_cosine_distribution

    """
    return rv(name, RaisedCosineDistribution, (mu, s))