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
@cacheit
def _taylor_term(self, n, x, *previous_terms):
    from sympy.functions.combinatorial.factorials import binomial
    return binomial(self.exp, n) * self.func(x, n)