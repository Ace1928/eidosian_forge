from __future__ import annotations
import re
import typing
from itertools import product
from typing import Any, Callable
import sympy
from sympy import Mul, Add, Pow, log, exp, sqrt, cos, sin, tan, asin, acos, acot, asec, acsc, sinh, cosh, tanh, asinh, \
from sympy.core.sympify import sympify, _sympify
from sympy.functions.special.bessel import airybiprime
from sympy.functions.special.error_functions import li
from sympy.utilities.exceptions import sympy_deprecation_warning
def _from_fullformlist_to_fullformsympy(self, pylist: list):
    from sympy import Function, Symbol

    def converter(expr):
        if isinstance(expr, list):
            if len(expr) > 0:
                head = expr[0]
                args = [converter(arg) for arg in expr[1:]]
                return Function(head)(*args)
            else:
                raise ValueError('Empty list of expressions')
        elif isinstance(expr, str):
            return Symbol(expr)
        else:
            return _sympify(expr)
    return converter(pylist)