from sympy.core.basic import Basic
from sympy.core.containers import (Dict, Tuple)
from sympy.core.expr import Expr
from sympy.core.kind import Kind, NumberKind, UndefinedKind
from sympy.core.numbers import Integer
from sympy.core.singleton import S
from sympy.core.sympify import sympify
from sympy.external.gmpy import SYMPY_INTS
from sympy.printing.defaults import Printable
import itertools
from collections.abc import Iterable
class ImmutableNDimArray(NDimArray, Basic):
    _op_priority = 11.0

    def __hash__(self):
        return Basic.__hash__(self)

    def as_immutable(self):
        return self

    def as_mutable(self):
        raise NotImplementedError('abstract method')