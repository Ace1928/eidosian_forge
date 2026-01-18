from sympy.core.sympify import _sympify
from sympy.core import S, Basic
from sympy.matrices.common import NonSquareMatrixError
from sympy.matrices.expressions.matpow import MatPow
from sympy.assumptions.ask import ask, Q
from sympy.assumptions.refine import handlers_dict
def _eval_determinant(self):
    from sympy.matrices.expressions.determinant import det
    return 1 / det(self.arg)