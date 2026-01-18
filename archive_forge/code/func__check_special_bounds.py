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
@classmethod
def _check_special_bounds(cls, flat_list, shape):
    if shape == () and len(flat_list) != 1:
        raise ValueError('arrays without shape need one scalar value')
    if shape == (0,) and len(flat_list) > 0:
        raise ValueError('if array shape is (0,) there cannot be elements')