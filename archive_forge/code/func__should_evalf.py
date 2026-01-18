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
@classmethod
def _should_evalf(cls, arg):
    """
        Decide if the function should automatically evalf().

        Explanation
        ===========

        By default (in this implementation), this happens if (and only if) the
        ARG is a floating point number (including complex numbers).
        This function is used by __new__.

        Returns the precision to evalf to, or -1 if it should not evalf.
        """
    if arg.is_Float:
        return arg._prec
    if not arg.is_Add:
        return -1
    m = pure_complex(arg)
    if m is None:
        return -1
    return max(m[0]._prec, m[1]._prec)