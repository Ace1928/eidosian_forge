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
def FisherZ(name, d1, d2):
    """
    Create a Continuous Random Variable with an Fisher's Z distribution.

    Explanation
    ===========

    The density of the Fisher's Z distribution is given by

    .. math::
        f(x) := \\frac{2d_1^{d_1/2} d_2^{d_2/2}} {\\mathrm{B}(d_1/2, d_2/2)}
                \\frac{e^{d_1z}}{\\left(d_1e^{2z}+d_2\\right)^{\\left(d_1+d_2\\right)/2}}


    .. TODO - What is the difference between these degrees of freedom?

    Parameters
    ==========

    d1 : `d_1 > 0`
        Degree of freedom.
    d2 : `d_2 > 0`
        Degree of freedom.

    Returns
    =======

    RandomSymbol

    Examples
    ========

    >>> from sympy.stats import FisherZ, density
    >>> from sympy import Symbol, pprint

    >>> d1 = Symbol("d1", positive=True)
    >>> d2 = Symbol("d2", positive=True)
    >>> z = Symbol("z")

    >>> X = FisherZ("x", d1, d2)

    >>> D = density(X)(z)
    >>> pprint(D, use_unicode=False)
                                d1   d2
        d1   d2               - -- - --
        --   --                 2    2
        2    2  /    2*z     \\           d1*z
    2*d1  *d2  *\\d1*e    + d2/         *e
    -----------------------------------------
                     /d1  d2\\
                    B|--, --|
                     \\2   2 /

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Fisher%27s_z-distribution
    .. [2] https://mathworld.wolfram.com/Fishersz-Distribution.html

    """
    return rv(name, FisherZDistribution, (d1, d2))