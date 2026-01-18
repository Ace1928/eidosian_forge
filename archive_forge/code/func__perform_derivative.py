from sympy import permutedims
from sympy.core.numbers import Number
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.core.sympify import sympify
from sympy.tensor.tensor import Tensor, TensExpr, TensAdd, TensMul
def _perform_derivative(self):
    result = self.expr
    for v in self.variables:
        if isinstance(result, TensExpr):
            result = result._eval_partial_derivative(v)
        elif v._diff_wrt:
            result = result._eval_derivative(v)
        else:
            result = S.Zero
    return result