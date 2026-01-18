from __future__ import annotations
from typing import Any
from sympy.external import import_module
from sympy.printing.printer import Printer
from sympy.utilities.iterables import is_sequence
import sympy
from functools import partial
from sympy.utilities.decorator import doctest_depends_on
from sympy.utilities.exceptions import sympy_deprecation_warning
def _print_Basic(self, expr, **kwargs):
    op = mapping[type(expr)]
    children = [self._print(arg, **kwargs) for arg in expr.args]
    return op(*children)