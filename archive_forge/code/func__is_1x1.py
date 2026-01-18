from sympy.assumptions.ask import ask, Q
from sympy.core.relational import Eq
from sympy.core.singleton import S
from sympy.core.sympify import _sympify
from sympy.functions.special.tensor_functions import KroneckerDelta
from sympy.matrices.common import NonInvertibleMatrixError
from .matexpr import MatrixExpr
def _is_1x1(self):
    """Returns true if the matrix is known to be 1x1"""
    shape = self.shape
    return Eq(shape[0], 1) & Eq(shape[1], 1)