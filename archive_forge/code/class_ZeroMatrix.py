from sympy.assumptions.ask import ask, Q
from sympy.core.relational import Eq
from sympy.core.singleton import S
from sympy.core.sympify import _sympify
from sympy.functions.special.tensor_functions import KroneckerDelta
from sympy.matrices.common import NonInvertibleMatrixError
from .matexpr import MatrixExpr
class ZeroMatrix(MatrixExpr):
    """The Matrix Zero 0 - additive identity

    Examples
    ========

    >>> from sympy import MatrixSymbol, ZeroMatrix
    >>> A = MatrixSymbol('A', 3, 5)
    >>> Z = ZeroMatrix(3, 5)
    >>> A + Z
    A
    >>> Z*A.T
    0
    """
    is_ZeroMatrix = True

    def __new__(cls, m, n):
        m, n = (_sympify(m), _sympify(n))
        cls._check_dim(m)
        cls._check_dim(n)
        return super().__new__(cls, m, n)

    @property
    def shape(self):
        return (self.args[0], self.args[1])

    def _eval_power(self, exp):
        if (exp < 0) == True:
            raise NonInvertibleMatrixError('Matrix det == 0; not invertible')
        return self

    def _eval_transpose(self):
        return ZeroMatrix(self.cols, self.rows)

    def _eval_adjoint(self):
        return ZeroMatrix(self.cols, self.rows)

    def _eval_trace(self):
        return S.Zero

    def _eval_determinant(self):
        return S.Zero

    def _eval_inverse(self):
        raise NonInvertibleMatrixError('Matrix det == 0; not invertible.')

    def _eval_as_real_imag(self):
        return (self, self)

    def _eval_conjugate(self):
        return self

    def _entry(self, i, j, **kwargs):
        return S.Zero