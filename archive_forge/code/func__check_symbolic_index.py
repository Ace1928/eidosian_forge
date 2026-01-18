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
def _check_symbolic_index(self, index):
    tuple_index = index if isinstance(index, tuple) else (index,)
    if any((isinstance(i, Expr) and (not i.is_number) for i in tuple_index)):
        for i, nth_dim in zip(tuple_index, self.shape):
            if (i < 0) == True or (i >= nth_dim) == True:
                raise ValueError('index out of range')
        from sympy.tensor import Indexed
        return Indexed(self, *tuple_index)
    return None