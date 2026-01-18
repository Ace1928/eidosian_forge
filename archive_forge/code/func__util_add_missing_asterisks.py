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
def _util_add_missing_asterisks(self, tokens: list):
    size: int = len(tokens)
    pointer: int = 0
    while pointer < size:
        if pointer > 0 and self._is_valid_star1(tokens[pointer - 1]) and self._is_valid_star2(tokens[pointer]):
            if tokens[pointer] == '(':
                tokens[pointer] = '*'
                tokens[pointer + 1] = tokens[pointer + 1][0]
            else:
                tokens.insert(pointer, '*')
                pointer += 1
                size += 1
        pointer += 1