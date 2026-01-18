from __future__ import annotations
from typing import Any
from sympy.external import import_module
from sympy.printing.printer import Printer
from sympy.utilities.iterables import is_sequence
import sympy
from functools import partial
from sympy.utilities.decorator import doctest_depends_on
from sympy.utilities.exceptions import sympy_deprecation_warning
def _print_slice(self, expr, **kwargs):
    return slice(*[self._print(i, **kwargs) if isinstance(i, sympy.Basic) else i for i in (expr.start, expr.stop, expr.step)])