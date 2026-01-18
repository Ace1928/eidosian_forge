from sympy.core.sympify import _sympify
from sympy.core import S, Basic
from sympy.matrices.common import NonSquareMatrixError
from sympy.matrices.expressions.matpow import MatPow
from sympy.assumptions.ask import ask, Q
from sympy.assumptions.refine import handlers_dict
class Inverse(MatPow):
    """
    The multiplicative inverse of a matrix expression

    This is a symbolic object that simply stores its argument without
    evaluating it. To actually compute the inverse, use the ``.inverse()``
    method of matrices.

    Examples
    ========

    >>> from sympy import MatrixSymbol, Inverse
    >>> A = MatrixSymbol('A', 3, 3)
    >>> B = MatrixSymbol('B', 3, 3)
    >>> Inverse(A)
    A**(-1)
    >>> A.inverse() == Inverse(A)
    True
    >>> (A*B).inverse()
    B**(-1)*A**(-1)
    >>> Inverse(A*B)
    (A*B)**(-1)

    """
    is_Inverse = True
    exp = S.NegativeOne

    def __new__(cls, mat, exp=S.NegativeOne):
        mat = _sympify(mat)
        exp = _sympify(exp)
        if not mat.is_Matrix:
            raise TypeError('mat should be a matrix')
        if mat.is_square is False:
            raise NonSquareMatrixError('Inverse of non-square matrix %s' % mat)
        return Basic.__new__(cls, mat, exp)

    @property
    def arg(self):
        return self.args[0]

    @property
    def shape(self):
        return self.arg.shape

    def _eval_inverse(self):
        return self.arg

    def _eval_determinant(self):
        from sympy.matrices.expressions.determinant import det
        return 1 / det(self.arg)

    def doit(self, **hints):
        if 'inv_expand' in hints and hints['inv_expand'] == False:
            return self
        arg = self.arg
        if hints.get('deep', True):
            arg = arg.doit(**hints)
        return arg.inverse()

    def _eval_derivative_matrix_lines(self, x):
        arg = self.args[0]
        lines = arg._eval_derivative_matrix_lines(x)
        for line in lines:
            line.first_pointer *= -self.T
            line.second_pointer *= self
        return lines