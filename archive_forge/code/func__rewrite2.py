from __future__ import annotations
import itertools
from sympy import SYMPY_DEBUG
from sympy.core import S, Expr
from sympy.core.add import Add
from sympy.core.basic import Basic
from sympy.core.cache import cacheit
from sympy.core.containers import Tuple
from sympy.core.exprtools import factor_terms
from sympy.core.function import (expand, expand_mul, expand_power_base,
from sympy.core.mul import Mul
from sympy.core.numbers import ilcm, Rational, pi
from sympy.core.relational import Eq, Ne, _canonical_coeff
from sympy.core.sorting import default_sort_key, ordered
from sympy.core.symbol import Dummy, symbols, Wild, Symbol
from sympy.core.sympify import sympify
from sympy.functions.combinatorial.factorials import factorial
from sympy.functions.elementary.complexes import (re, im, arg, Abs, sign,
from sympy.functions.elementary.exponential import exp, exp_polar, log
from sympy.functions.elementary.integers import ceiling
from sympy.functions.elementary.hyperbolic import (cosh, sinh,
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.piecewise import Piecewise, piecewise_fold
from sympy.functions.elementary.trigonometric import (cos, sin, sinc,
from sympy.functions.special.bessel import besselj, bessely, besseli, besselk
from sympy.functions.special.delta_functions import DiracDelta, Heaviside
from sympy.functions.special.elliptic_integrals import elliptic_k, elliptic_e
from sympy.functions.special.error_functions import (erf, erfc, erfi, Ei,
from sympy.functions.special.gamma_functions import gamma
from sympy.functions.special.hyper import hyper, meijerg
from sympy.functions.special.singularity_functions import SingularityFunction
from .integrals import Integral
from sympy.logic.boolalg import And, Or, BooleanAtom, Not, BooleanFunction
from sympy.polys import cancel, factor
from sympy.utilities.iterables import multiset_partitions
from sympy.utilities.misc import debug as _debug
from sympy.utilities.misc import debugf as _debugf
from sympy.utilities.timeutils import timethis
def _rewrite2(f, x):
    """
    Try to rewrite ``f`` as a product of two G functions of arguments a*x**b.
    Return fac, po, g1, g2 such that f = fac*po*g1*g2, where fac is
    independent of x and po is x**s.
    Here g1 and g2 are results of _rewrite_single.
    Returns None on failure.
    """
    fac, po, g = _split_mul(f, x)
    if any((_rewrite_single(expr, x, False) is None for expr in _mul_args(g))):
        return None
    l = _mul_as_two_parts(g)
    if not l:
        return None
    l = list(ordered(l, [lambda p: max(len(_exponents(p[0], x)), len(_exponents(p[1], x))), lambda p: max(len(_functions(p[0], x)), len(_functions(p[1], x))), lambda p: max(len(_find_splitting_points(p[0], x)), len(_find_splitting_points(p[1], x)))]))
    for recursive, (fac1, fac2) in itertools.product((False, True), l):
        g1 = _rewrite_single(fac1, x, recursive)
        g2 = _rewrite_single(fac2, x, recursive)
        if g1 and g2:
            cond = And(g1[1], g2[1])
            if cond != False:
                return (fac, po, g1[0], g2[0], cond)