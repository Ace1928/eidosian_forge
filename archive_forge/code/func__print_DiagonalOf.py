from sympy.core import S
from sympy.core.function import Lambda
from sympy.core.power import Pow
from .pycode import PythonCodePrinter, _known_functions_math, _print_known_const, _print_known_func, _unpack_integral_limits, ArrayPrinter
from .codeprinter import CodePrinter
def _print_DiagonalOf(self, expr):
    vect = '{}({})'.format(self._module_format(self._module + '.diag'), self._print(expr.arg))
    return '{}({}, (-1, 1))'.format(self._module_format(self._module + '.reshape'), vect)