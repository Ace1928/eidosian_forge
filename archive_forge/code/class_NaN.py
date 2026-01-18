from __future__ import annotations
import numbers
import decimal
import fractions
import math
import re as regex
import sys
from functools import lru_cache
from .containers import Tuple
from .sympify import (SympifyError, _sympy_converter, sympify, _convert_numpy_types,
from .singleton import S, Singleton
from .basic import Basic
from .expr import Expr, AtomicExpr
from .evalf import pure_complex
from .cache import cacheit, clear_cache
from .decorators import _sympifyit
from .logic import fuzzy_not
from .kind import NumberKind
from sympy.external.gmpy import SYMPY_INTS, HAS_GMPY, gmpy
from sympy.multipledispatch import dispatch
import mpmath
import mpmath.libmp as mlib
from mpmath.libmp import bitcount, round_nearest as rnd
from mpmath.libmp.backend import MPZ
from mpmath.libmp import mpf_pow, mpf_pi, mpf_e, phi_fixed
from mpmath.ctx_mp import mpnumeric
from mpmath.libmp.libmpf import (
from sympy.utilities.misc import as_int, debug, filldedent
from .parameters import global_parameters
from .power import Pow, integer_nthroot
from .mul import Mul
from .add import Add
class NaN(Number, metaclass=Singleton):
    """
    Not a Number.

    Explanation
    ===========

    This serves as a place holder for numeric values that are indeterminate.
    Most operations on NaN, produce another NaN.  Most indeterminate forms,
    such as ``0/0`` or ``oo - oo` produce NaN.  Two exceptions are ``0**0``
    and ``oo**0``, which all produce ``1`` (this is consistent with Python's
    float).

    NaN is loosely related to floating point nan, which is defined in the
    IEEE 754 floating point standard, and corresponds to the Python
    ``float('nan')``.  Differences are noted below.

    NaN is mathematically not equal to anything else, even NaN itself.  This
    explains the initially counter-intuitive results with ``Eq`` and ``==`` in
    the examples below.

    NaN is not comparable so inequalities raise a TypeError.  This is in
    contrast with floating point nan where all inequalities are false.

    NaN is a singleton, and can be accessed by ``S.NaN``, or can be imported
    as ``nan``.

    Examples
    ========

    >>> from sympy import nan, S, oo, Eq
    >>> nan is S.NaN
    True
    >>> oo - oo
    nan
    >>> nan + 1
    nan
    >>> Eq(nan, nan)   # mathematical equality
    False
    >>> nan == nan     # structural equality
    True

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/NaN

    """
    is_commutative = True
    is_extended_real = None
    is_real = None
    is_rational = None
    is_algebraic = None
    is_transcendental = None
    is_integer = None
    is_comparable = False
    is_finite = None
    is_zero = None
    is_prime = None
    is_positive = None
    is_negative = None
    is_number = True
    __slots__ = ()

    def __new__(cls):
        return AtomicExpr.__new__(cls)

    def _latex(self, printer):
        return '\\text{NaN}'

    def __neg__(self):
        return self

    @_sympifyit('other', NotImplemented)
    def __add__(self, other):
        return self

    @_sympifyit('other', NotImplemented)
    def __sub__(self, other):
        return self

    @_sympifyit('other', NotImplemented)
    def __mul__(self, other):
        return self

    @_sympifyit('other', NotImplemented)
    def __truediv__(self, other):
        return self

    def floor(self):
        return self

    def ceiling(self):
        return self

    def _as_mpf_val(self, prec):
        return _mpf_nan

    def __hash__(self):
        return super().__hash__()

    def __eq__(self, other):
        return other is S.NaN

    def __ne__(self, other):
        return other is not S.NaN
    __gt__ = Expr.__gt__
    __ge__ = Expr.__ge__
    __lt__ = Expr.__lt__
    __le__ = Expr.__le__