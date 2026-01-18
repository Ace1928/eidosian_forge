import re
from sympy.core.numbers import (I, NumberSymbol, oo, zoo)
from sympy.core.symbol import Symbol
from sympy.utilities.iterables import numbered_symbols
from sympy.external import import_module
import warnings
class lambdify:
    """Returns the lambdified function.

    Explanation
    ===========

    This function uses experimental_lambdify to create a lambdified
    expression. It uses cmath to lambdify the expression. If the function
    is not implemented in Python cmath, Python cmath calls evalf on those
    functions.
    """

    def __init__(self, args, expr):
        self.args = args
        self.expr = expr
        self.lambda_func_1 = experimental_lambdify(args, expr, use_python_cmath=True, use_evalf=True)
        self.lambda_func_2 = experimental_lambdify(args, expr, use_python_math=True, use_evalf=True)
        self.lambda_func_3 = experimental_lambdify(args, expr, use_evalf=True, complex_wrap_evalf=True)
        self.lambda_func = self.lambda_func_1
        self.failure = False

    def __call__(self, args):
        try:
            result = complex(self.lambda_func(args))
            if abs(result.imag) > 1e-07 * abs(result):
                return None
            return result.real
        except (ZeroDivisionError, OverflowError):
            return None
        except TypeError as e:
            if self.failure:
                raise e
            if self.lambda_func == self.lambda_func_1:
                self.lambda_func = self.lambda_func_2
                return self.__call__(args)
            self.failure = True
            self.lambda_func = self.lambda_func_3
            warnings.warn('The evaluation of the expression is problematic. We are trying a failback method that may still work. Please report this as a bug.', stacklevel=2)
            return self.__call__(args)