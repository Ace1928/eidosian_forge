from functools import reduce, wraps
from itertools import repeat
from sympy.core import S, pi
from sympy.core.add import Add
from sympy.core.function import (
from sympy.core.mul import Mul
from sympy.core.numbers import igcd, ilcm
from sympy.core.sorting import default_sort_key
from sympy.core.symbol import Dummy
from sympy.core.traversal import postorder_traversal
from sympy.functions.combinatorial.factorials import factorial, rf
from sympy.functions.elementary.complexes import re, arg, Abs
from sympy.functions.elementary.exponential import exp, exp_polar
from sympy.functions.elementary.hyperbolic import cosh, coth, sinh, tanh
from sympy.functions.elementary.integers import ceiling
from sympy.functions.elementary.miscellaneous import Max, Min, sqrt
from sympy.functions.elementary.piecewise import piecewise_fold
from sympy.functions.elementary.trigonometric import cos, cot, sin, tan
from sympy.functions.special.bessel import besselj
from sympy.functions.special.delta_functions import Heaviside
from sympy.functions.special.gamma_functions import gamma
from sympy.functions.special.hyper import meijerg
from sympy.integrals import integrate, Integral
from sympy.integrals.meijerint import _dummy
from sympy.logic.boolalg import to_cnf, conjuncts, disjuncts, Or, And
from sympy.polys.polyroots import roots
from sympy.polys.polytools import factor, Poly
from sympy.polys.rootoftools import CRootOf
from sympy.utilities.iterables import iterable
from sympy.utilities.misc import debug
import sympy.integrals.laplace as _laplace
@_noconds_(True)
def _fourier_transform(f, x, k, a, b, name, simplify=True):
    """
    Compute a general Fourier-type transform

    .. math::

        F(k) = a \\int_{-\\infty}^{\\infty} e^{bixk} f(x)\\, dx.

    For suitable choice of *a* and *b*, this reduces to the standard Fourier
    and inverse Fourier transforms.
    """
    F = integrate(a * f * exp(b * S.ImaginaryUnit * x * k), (x, S.NegativeInfinity, S.Infinity))
    if not F.has(Integral):
        return (_simplify(F, simplify), S.true)
    integral_f = integrate(f, (x, S.NegativeInfinity, S.Infinity))
    if integral_f in (S.NegativeInfinity, S.Infinity, S.NaN) or integral_f.has(Integral):
        raise IntegralTransformError(name, f, 'function not integrable on real axis')
    if not F.is_Piecewise:
        raise IntegralTransformError(name, f, 'could not compute integral')
    F, cond = F.args[0]
    if F.has(Integral):
        raise IntegralTransformError(name, f, 'integral in unexpected form')
    return (_simplify(F, simplify), cond)