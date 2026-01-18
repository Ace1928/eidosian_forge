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
def _eval_is_nonnegative(self):
    return self.args[0].is_nonnegative