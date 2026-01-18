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
def _from_fullformlist_to_sympy(self, full_form_list):

    def recurse(expr):
        if isinstance(expr, list):
            if isinstance(expr[0], list):
                head = recurse(expr[0])
            else:
                head = self._node_conversions.get(expr[0], Function(expr[0]))
            return head(*[recurse(arg) for arg in expr[1:]])
        else:
            return self._atom_conversions.get(expr, sympify(expr))
    return recurse(full_form_list)