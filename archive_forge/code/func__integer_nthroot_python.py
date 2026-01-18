from __future__ import annotations
from typing import Callable
from math import log as _log, sqrt as _sqrt
from itertools import product
from .sympify import _sympify
from .cache import cacheit
from .singleton import S
from .expr import Expr
from .evalf import PrecisionExhausted
from .function import (expand_complex, expand_multinomial,
from .logic import fuzzy_bool, fuzzy_not, fuzzy_and, fuzzy_or
from .parameters import global_parameters
from .relational import is_gt, is_lt
from .kind import NumberKind, UndefinedKind
from sympy.external.gmpy import HAS_GMPY, gmpy
from sympy.utilities.iterables import sift
from sympy.utilities.exceptions import sympy_deprecation_warning
from sympy.utilities.misc import as_int
from sympy.multipledispatch import Dispatcher
from mpmath.libmp import sqrtrem as mpmath_sqrtrem
from .add import Add
from .numbers import Integer
from .mul import Mul, _keep_coeff
from .symbol import Symbol, Dummy, symbols
def _integer_nthroot_python(y, n):
    if y in (0, 1):
        return (y, True)
    if n == 1:
        return (y, True)
    if n == 2:
        x, rem = mpmath_sqrtrem(y)
        return (int(x), not rem)
    if n >= y.bit_length():
        return (1, False)
    try:
        guess = int(y ** (1.0 / n) + 0.5)
    except OverflowError:
        exp = _log(y, 2) / n
        if exp > 53:
            shift = int(exp - 53)
            guess = int(2.0 ** (exp - shift) + 1) << shift
        else:
            guess = int(2.0 ** exp)
    if guess > 2 ** 50:
        xprev, x = (-1, guess)
        while 1:
            t = x ** (n - 1)
            xprev, x = (x, ((n - 1) * x + y // t) // n)
            if abs(x - xprev) < 2:
                break
    else:
        x = guess
    t = x ** n
    while t < y:
        x += 1
        t = x ** n
    while t > y:
        x -= 1
        t = x ** n
    return (int(x), t == y)