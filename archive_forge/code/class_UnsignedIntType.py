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
class UnsignedIntType(_SizedIntType):
    """ Represents an unsigned integer type. """
    __slots__ = ()

    @property
    def min(self):
        return 0

    @property
    def max(self):
        return 2 ** self.nbits - 1