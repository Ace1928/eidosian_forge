from __future__ import annotations
from typing import Any
from collections import defaultdict
from sympy.core.relational import (Ge, Gt, Le, Lt)
from sympy.core import Symbol, Tuple, Dummy
from sympy.core.basic import Basic
from sympy.core.expr import Expr, Atom
from sympy.core.numbers import Float, Integer, oo
from sympy.core.sympify import _sympify, sympify, SympifyError
from sympy.utilities.iterables import (iterable, topological_sort,
class _SizedIntType(IntBaseType):
    __slots__ = ('nbits',)
    _fields = Type._fields + __slots__
    _construct_nbits = Integer

    def _check(self, value):
        if value < self.min:
            raise ValueError('Value is too small: %d < %d' % (value, self.min))
        if value > self.max:
            raise ValueError('Value is too big: %d > %d' % (value, self.max))