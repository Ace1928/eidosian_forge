import re
from sympy.core.numbers import (I, NumberSymbol, oo, zoo)
from sympy.core.symbol import Symbol
from sympy.utilities.iterables import numbered_symbols
from sympy.external import import_module
import warnings
class vectorized_lambdify:
    """ Return a sufficiently smart, vectorized and lambdified function.

    Returns only reals.

    Explanation
    ===========

    This function uses experimental_lambdify to created a lambdified
    expression ready to be used with numpy. Many of the functions in SymPy
    are not implemented in numpy so in some cases we resort to Python cmath or
    even to evalf.

    The following translations are tried:
      only numpy complex
      - on errors raised by SymPy trying to work with ndarray:
          only Python cmath and then vectorize complex128

    When using Python cmath there is no need for evalf or float/complex
    because Python cmath calls those.

    This function never tries to mix numpy directly with evalf because numpy
    does not understand SymPy Float. If this is needed one can use the
    float_wrap_evalf/complex_wrap_evalf options of experimental_lambdify or
    better one can be explicit about the dtypes that numpy works with.
    Check numpy bug http://projects.scipy.org/numpy/ticket/1013 to know what
    types of errors to expect.
    """

    def __init__(self, args, expr):
        self.args = args
        self.expr = expr
        self.np = import_module('numpy')
        self.lambda_func_1 = experimental_lambdify(args, expr, use_np=True)
        self.vector_func_1 = self.lambda_func_1
        self.lambda_func_2 = experimental_lambdify(args, expr, use_python_cmath=True)
        self.vector_func_2 = self.np.vectorize(self.lambda_func_2, otypes=[complex])
        self.vector_func = self.vector_func_1
        self.failure = False

    def __call__(self, *args):
        np = self.np
        try:
            temp_args = (np.array(a, dtype=complex) for a in args)
            results = self.vector_func(*temp_args)
            results = np.ma.masked_where(np.abs(results.imag) > 1e-07 * np.abs(results), results.real, copy=False)
            return results
        except ValueError:
            if self.failure:
                raise
            self.failure = True
            self.vector_func = self.vector_func_2
            warnings.warn('The evaluation of the expression is problematic. We are trying a failback method that may still work. Please report this as a bug.')
            return self.__call__(*args)