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
def _util_remove_newlines(self, lines: list, tokens: list, inside_enclosure: bool):
    pointer = 0
    size = len(tokens)
    while pointer < size:
        token = tokens[pointer]
        if token == '\n':
            if inside_enclosure:
                tokens.pop(pointer)
                size -= 1
                continue
            if pointer == 0:
                tokens.pop(0)
                size -= 1
                continue
            if pointer > 1:
                try:
                    prev_expr = self._parse_after_braces(tokens[:pointer], inside_enclosure)
                except SyntaxError:
                    tokens.pop(pointer)
                    size -= 1
                    continue
            else:
                prev_expr = tokens[0]
            if len(prev_expr) > 0 and prev_expr[0] == 'CompoundExpression':
                lines.extend(prev_expr[1:])
            else:
                lines.append(prev_expr)
            for i in range(pointer):
                tokens.pop(0)
            size -= pointer
            pointer = 0
            continue
        pointer += 1