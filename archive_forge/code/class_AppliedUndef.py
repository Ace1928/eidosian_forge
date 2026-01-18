from __future__ import annotations
from typing import Any
from collections.abc import Iterable
from .add import Add
from .basic import Basic, _atomic
from .cache import cacheit
from .containers import Tuple, Dict
from .decorators import _sympifyit
from .evalf import pure_complex
from .expr import Expr, AtomicExpr
from .logic import fuzzy_and, fuzzy_or, fuzzy_not, FuzzyBool
from .mul import Mul
from .numbers import Rational, Float, Integer
from .operations import LatticeOp
from .parameters import global_parameters
from .rules import Transform
from .singleton import S
from .sympify import sympify, _sympify
from .sorting import default_sort_key, ordered
from sympy.utilities.exceptions import (sympy_deprecation_warning,
from sympy.utilities.iterables import (has_dups, sift, iterable,
from sympy.utilities.lambdify import MPMATH_TRANSLATIONS
from sympy.utilities.misc import as_int, filldedent, func_name
import mpmath
from mpmath.libmp.libmpf import prec_to_dps
import inspect
from collections import Counter
from .symbol import Dummy, Symbol
class AppliedUndef(Function):
    """
    Base class for expressions resulting from the application of an undefined
    function.
    """
    is_number = False

    def __new__(cls, *args, **options):
        args = list(map(sympify, args))
        u = [a.name for a in args if isinstance(a, UndefinedFunction)]
        if u:
            raise TypeError('Invalid argument: expecting an expression, not UndefinedFunction%s: %s' % ('s' * (len(u) > 1), ', '.join(u)))
        obj = super().__new__(cls, *args, **options)
        return obj

    def _eval_as_leading_term(self, x, logx=None, cdir=0):
        return self

    @property
    def _diff_wrt(self):
        """
        Allow derivatives wrt to undefined functions.

        Examples
        ========

        >>> from sympy import Function, Symbol
        >>> f = Function('f')
        >>> x = Symbol('x')
        >>> f(x)._diff_wrt
        True
        >>> f(x).diff(x)
        Derivative(f(x), x)
        """
        return True