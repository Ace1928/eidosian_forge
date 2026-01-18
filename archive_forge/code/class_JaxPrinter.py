from sympy.core import S
from sympy.core.function import Lambda
from sympy.core.power import Pow
from .pycode import PythonCodePrinter, _known_functions_math, _print_known_const, _print_known_func, _unpack_integral_limits, ArrayPrinter
from .codeprinter import CodePrinter
class JaxPrinter(NumPyPrinter):
    """
    JAX printer which handles vectorized piecewise functions,
    logical operators, etc.
    """
    _module = 'jax.numpy'
    _kf = _jax_known_functions
    _kc = _jax_known_constants

    def __init__(self, settings=None):
        super().__init__(settings=settings)

    def _print_And(self, expr):
        """Logical And printer"""
        return '{}({}.asarray([{}]), axis=0)'.format(self._module_format(self._module + '.all'), self._module_format(self._module), ','.join((self._print(i) for i in expr.args)))

    def _print_Or(self, expr):
        """Logical Or printer"""
        return '{}({}.asarray([{}]), axis=0)'.format(self._module_format(self._module + '.any'), self._module_format(self._module), ','.join((self._print(i) for i in expr.args)))