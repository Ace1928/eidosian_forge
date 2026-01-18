from sympy.core import S
from sympy.core.function import Lambda
from sympy.core.power import Pow
from .pycode import PythonCodePrinter, _known_functions_math, _print_known_const, _print_known_func, _unpack_integral_limits, ArrayPrinter
from .codeprinter import CodePrinter
def _print_DiagonalMatrix(self, expr):
    return '{}({}, {}({}, {}))'.format(self._module_format(self._module + '.multiply'), self._print(expr.arg), self._module_format(self._module + '.eye'), self._print(expr.shape[0]), self._print(expr.shape[1]))