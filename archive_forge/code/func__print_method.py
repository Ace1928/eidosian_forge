from itertools import chain
from sympy.codegen.ast import Type, none
from .c import C89CodePrinter, C99CodePrinter
from sympy.printing.codeprinter import cxxcode # noqa:F401
def _print_method(self, expr):
    return '{}{}({})'.format(self._ns, func_name, ', '.join(map(self._print, expr.args)))