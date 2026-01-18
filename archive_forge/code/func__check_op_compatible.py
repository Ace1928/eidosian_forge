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
def _check_op_compatible(self, op1: str, op2: str):
    if op1 == op2:
        return True
    muldiv = {'*', '/'}
    addsub = {'+', '-'}
    if op1 in muldiv and op2 in muldiv:
        return True
    if op1 in addsub and op2 in addsub:
        return True
    return False