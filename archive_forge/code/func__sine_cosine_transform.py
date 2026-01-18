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
def _sine_cosine_transform(f, x, k, a, b, K, name, simplify=True):
    """
    Compute a general sine or cosine-type transform
        F(k) = a int_0^oo b*sin(x*k) f(x) dx.
        F(k) = a int_0^oo b*cos(x*k) f(x) dx.

    For suitable choice of a and b, this reduces to the standard sine/cosine
    and inverse sine/cosine transforms.
    """
    F = integrate(a * f * K(b * x * k), (x, S.Zero, S.Infinity))
    if not F.has(Integral):
        return (_simplify(F, simplify), S.true)
    if not F.is_Piecewise:
        raise IntegralTransformError(name, f, 'could not compute integral')
    F, cond = F.args[0]
    if F.has(Integral):
        raise IntegralTransformError(name, f, 'integral in unexpected form')
    return (_simplify(F, simplify), cond)