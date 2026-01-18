from __future__ import annotations
from typing import Any
from functools import wraps
from itertools import chain
from sympy.core import S
from sympy.core.numbers import equal_valued
from sympy.codegen.ast import (
from sympy.printing.codeprinter import CodePrinter, requires
from sympy.printing.precedence import precedence, PRECEDENCE
from sympy.sets.fancysets import Range
from sympy.printing.codeprinter import ccode, print_ccode # noqa:F401
def _print_struct(self, expr):
    return '%(keyword)s %(name)s {\n%(lines)s}' % {'keyword': expr.__class__.__name__, 'name': expr.name, 'lines': ';\n'.join([self._print(decl) for decl in expr.declarations] + [''])}