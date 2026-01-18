from sympy.assumptions.ask import ask, Q
from sympy.core.relational import Eq
from sympy.core.singleton import S
from sympy.core.sympify import _sympify
from sympy.functions.special.tensor_functions import KroneckerDelta
from sympy.matrices.common import NonInvertibleMatrixError
from .matexpr import MatrixExpr
class OneMatrix(MatrixExpr):
    """
    Matrix whose all entries are ones.
    """

    def __new__(cls, m, n, evaluate=False):
        m, n = (_sympify(m), _sympify(n))
        cls._check_dim(m)
        cls._check_dim(n)
        if evaluate:
            condition = Eq(m, 1) & Eq(n, 1)
            if condition == True:
                return Identity(1)
        obj = super().__new__(cls, m, n)
        return obj

    @property
    def shape(self):
        return self._args

    @property
    def is_Identity(self):
        return self._is_1x1() == True

    def as_explicit(self):
        from sympy.matrices.immutable import ImmutableDenseMatrix
        return ImmutableDenseMatrix.ones(*self.shape)

    def doit(self, **hints):
        args = self.args
        if hints.get('deep', True):
            args = [a.doit(**hints) for a in args]
        return self.func(*args, evaluate=True)

    def _eval_power(self, exp):
        if self._is_1x1() == True:
            return Identity(1)
        if (exp < 0) == True:
            raise NonInvertibleMatrixError('Matrix det == 0; not invertible')
        if ask(Q.integer(exp)):
            return self.shape[0] ** (exp - 1) * OneMatrix(*self.shape)
        return super()._eval_power(exp)

    def _eval_transpose(self):
        return OneMatrix(self.cols, self.rows)

    def _eval_adjoint(self):
        return OneMatrix(self.cols, self.rows)

    def _eval_trace(self):
        return S.One * self.rows

    def _is_1x1(self):
        """Returns true if the matrix is known to be 1x1"""
        shape = self.shape
        return Eq(shape[0], 1) & Eq(shape[1], 1)

    def _eval_determinant(self):
        condition = self._is_1x1()
        if condition == True:
            return S.One
        elif condition == False:
            return S.Zero
        else:
            from sympy.matrices.expressions.determinant import Determinant
            return Determinant(self)

    def _eval_inverse(self):
        condition = self._is_1x1()
        if condition == True:
            return Identity(1)
        elif condition == False:
            raise NonInvertibleMatrixError('Matrix det == 0; not invertible.')
        else:
            from .inverse import Inverse
            return Inverse(self)

    def _eval_as_real_imag(self):
        return (self, ZeroMatrix(*self.shape))

    def _eval_conjugate(self):
        return self

    def _entry(self, i, j, **kwargs):
        return S.One