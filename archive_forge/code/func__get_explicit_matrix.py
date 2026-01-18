from .matexpr import MatrixExpr
from .special import Identity
from sympy.core import S
from sympy.core.expr import ExprBuilder
from sympy.core.cache import cacheit
from sympy.core.power import Pow
from sympy.core.sympify import _sympify
from sympy.matrices import MatrixBase
from sympy.matrices.common import NonSquareMatrixError
@cacheit
def _get_explicit_matrix(self):
    return self.base.as_explicit() ** self.exp