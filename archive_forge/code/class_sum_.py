from sympy.codegen.ast import (
from sympy.core.basic import Basic
from sympy.core.containers import Tuple
from sympy.core.expr import Expr
from sympy.core.function import Function
from sympy.core.numbers import Float, Integer
from sympy.core.symbol import Str
from sympy.core.sympify import sympify
from sympy.logic import true, false
from sympy.utilities.iterables import iterable
class sum_(Token, Expr):
    __slots__ = _fields = ('array', 'dim', 'mask')
    defaults = {'dim': none, 'mask': none}
    _construct_array = staticmethod(sympify)
    _construct_dim = staticmethod(sympify)