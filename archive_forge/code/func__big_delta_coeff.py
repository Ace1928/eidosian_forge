from sympy.concrete.summations import Sum
from sympy.core.add import Add
from sympy.core.function import Function
from sympy.core.numbers import (I, Integer, pi)
from sympy.core.singleton import S
from sympy.core.symbol import Dummy
from sympy.core.sympify import sympify
from sympy.functions.combinatorial.factorials import (binomial, factorial)
from sympy.functions.elementary.complexes import re
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.functions.special.spherical_harmonics import Ynm
from sympy.matrices.dense import zeros
from sympy.matrices.immutable import ImmutableMatrix
from sympy.utilities.misc import as_int
def _big_delta_coeff(aa, bb, cc, prec=None):
    """
    Calculates the Delta coefficient of the 3 angular momenta for
    Racah symbols. Also checks that the differences are of integer
    value.

    Parameters
    ==========

    aa :
        First angular momentum, integer or half integer.
    bb :
        Second angular momentum, integer or half integer.
    cc :
        Third angular momentum, integer or half integer.
    prec :
        Precision of the ``sqrt()`` calculation.

    Returns
    =======

    double : Value of the Delta coefficient.

    Examples
    ========

        sage: from sage.functions.wigner import _big_delta_coeff
        sage: _big_delta_coeff(1,1,1)
        1/2*sqrt(1/6)
    """
    if int(aa + bb - cc) != aa + bb - cc:
        raise ValueError('j values must be integer or half integer and fulfill the triangle relation')
    if int(aa + cc - bb) != aa + cc - bb:
        raise ValueError('j values must be integer or half integer and fulfill the triangle relation')
    if int(bb + cc - aa) != bb + cc - aa:
        raise ValueError('j values must be integer or half integer and fulfill the triangle relation')
    if aa + bb - cc < 0:
        return S.Zero
    if aa + cc - bb < 0:
        return S.Zero
    if bb + cc - aa < 0:
        return S.Zero
    maxfact = max(aa + bb - cc, aa + cc - bb, bb + cc - aa, aa + bb + cc + 1)
    _calc_factlist(maxfact)
    argsqrt = Integer(_Factlist[int(aa + bb - cc)] * _Factlist[int(aa + cc - bb)] * _Factlist[int(bb + cc - aa)]) / Integer(_Factlist[int(aa + bb + cc + 1)])
    ressqrt = sqrt(argsqrt)
    if prec:
        ressqrt = ressqrt.evalf(prec).as_real_imag()[0]
    return ressqrt