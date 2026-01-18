from collections import defaultdict
from functools import reduce
from sympy.core import (sympify, Basic, S, Expr, factor_terms,
from sympy.core.cache import cacheit
from sympy.core.function import (count_ops, _mexpand, FunctionClass, expand,
from sympy.core.numbers import I, Integer, igcd
from sympy.core.sorting import _nodes
from sympy.core.symbol import Dummy, symbols, Wild
from sympy.external.gmpy import SYMPY_INTS
from sympy.functions import sin, cos, exp, cosh, tanh, sinh, tan, cot, coth
from sympy.functions import atan2
from sympy.functions.elementary.hyperbolic import HyperbolicFunction
from sympy.functions.elementary.trigonometric import TrigonometricFunction
from sympy.polys import Poly, factor, cancel, parallel_poly_from_expr
from sympy.polys.domains import ZZ
from sympy.polys.polyerrors import PolificationFailed
from sympy.polys.polytools import groebner
from sympy.simplify.cse_main import cse
from sympy.strategies.core import identity
from sympy.strategies.tree import greedy
from sympy.utilities.iterables import iterable
from sympy.utilities.misc import debug
def _match_div_rewrite(expr, i):
    """helper for __trigsimp"""
    if i == 0:
        expr = _replace_mul_fpowxgpow(expr, sin, cos, _midn, tan, _idn)
    elif i == 1:
        expr = _replace_mul_fpowxgpow(expr, tan, cos, _idn, sin, _idn)
    elif i == 2:
        expr = _replace_mul_fpowxgpow(expr, cot, sin, _idn, cos, _idn)
    elif i == 3:
        expr = _replace_mul_fpowxgpow(expr, tan, sin, _midn, cos, _midn)
    elif i == 4:
        expr = _replace_mul_fpowxgpow(expr, cot, cos, _midn, sin, _midn)
    elif i == 5:
        expr = _replace_mul_fpowxgpow(expr, cot, tan, _idn, _one, _idn)
    elif i == 8:
        expr = _replace_mul_fpowxgpow(expr, sinh, cosh, _midn, tanh, _idn)
    elif i == 9:
        expr = _replace_mul_fpowxgpow(expr, tanh, cosh, _idn, sinh, _idn)
    elif i == 10:
        expr = _replace_mul_fpowxgpow(expr, coth, sinh, _idn, cosh, _idn)
    elif i == 11:
        expr = _replace_mul_fpowxgpow(expr, tanh, sinh, _midn, cosh, _midn)
    elif i == 12:
        expr = _replace_mul_fpowxgpow(expr, coth, cosh, _midn, sinh, _midn)
    elif i == 13:
        expr = _replace_mul_fpowxgpow(expr, coth, tanh, _idn, _one, _idn)
    else:
        return None
    return expr