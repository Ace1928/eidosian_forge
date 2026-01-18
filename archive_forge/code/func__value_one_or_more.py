from typing import Tuple as tTuple
from sympy.core.basic import Basic
from sympy.core.expr import Expr
from sympy.core import Add, S
from sympy.core.evalf import get_integer_part, PrecisionExhausted
from sympy.core.function import Function
from sympy.core.logic import fuzzy_or
from sympy.core.numbers import Integer
from sympy.core.relational import Gt, Lt, Ge, Le, Relational, is_eq
from sympy.core.symbol import Symbol
from sympy.core.sympify import _sympify
from sympy.functions.elementary.complexes import im, re
from sympy.multipledispatch import dispatch
def _value_one_or_more(self, other):
    if other.is_extended_real:
        if other.is_number:
            res = other >= 1
            if res and (not isinstance(res, Relational)):
                return S.true
        if other.is_integer and other.is_positive:
            return S.true