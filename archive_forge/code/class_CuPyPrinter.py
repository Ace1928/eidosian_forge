from sympy.core import S
from sympy.core.function import Lambda
from sympy.core.power import Pow
from .pycode import PythonCodePrinter, _known_functions_math, _print_known_const, _print_known_func, _unpack_integral_limits, ArrayPrinter
from .codeprinter import CodePrinter
class CuPyPrinter(NumPyPrinter):
    """
    CuPy printer which handles vectorized piecewise functions,
    logical operators, etc.
    """
    _module = 'cupy'
    _kf = _cupy_known_functions
    _kc = _cupy_known_constants

    def __init__(self, settings=None):
        super().__init__(settings=settings)