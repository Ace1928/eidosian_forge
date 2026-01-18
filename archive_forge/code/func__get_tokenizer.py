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
def _get_tokenizer(self):
    if self._regex_tokenizer is not None:
        return self._regex_tokenizer
    tokens = [self._literal, self._number]
    tokens_escape = self._enclosure_open[:] + self._enclosure_close[:]
    for typ, strat, symdict in self._mathematica_op_precedence:
        for k in symdict:
            tokens_escape.append(k)
    tokens_escape.sort(key=lambda x: -len(x))
    tokens.extend(map(re.escape, tokens_escape))
    tokens.append(',')
    tokens.append('\n')
    tokenizer = re.compile('(' + '|'.join(tokens) + ')')
    self._regex_tokenizer = tokenizer
    return self._regex_tokenizer