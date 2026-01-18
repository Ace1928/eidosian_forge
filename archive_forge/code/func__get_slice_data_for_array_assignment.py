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
def _get_slice_data_for_array_assignment(self, index, value):
    if not isinstance(value, NDimArray):
        value = type(self)(value)
    sl_factors, eindices = self._get_slice_data_for_array_access(index)
    slice_offsets = [min(i) if isinstance(i, list) else None for i in sl_factors]
    return (value, eindices, slice_offsets)