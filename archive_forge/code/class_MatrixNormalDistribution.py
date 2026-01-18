from math import prod
from sympy.core.basic import Basic
from sympy.core.numbers import pi
from sympy.core.singleton import S
from sympy.functions.elementary.exponential import exp
from sympy.functions.special.gamma_functions import multigamma
from sympy.core.sympify import sympify, _sympify
from sympy.matrices import (ImmutableMatrix, Inverse, Trace, Determinant,
from sympy.stats.rv import (_value_check, RandomMatrixSymbol, NamedArgsMixin, PSpace,
from sympy.external import import_module
class MatrixNormalDistribution(MatrixDistribution):
    _argnames = ('location_matrix', 'scale_matrix_1', 'scale_matrix_2')

    @staticmethod
    def check(location_matrix, scale_matrix_1, scale_matrix_2):
        if not isinstance(scale_matrix_1, MatrixSymbol):
            _value_check(scale_matrix_1.is_positive_definite, 'The shape matrix must be positive definite.')
        if not isinstance(scale_matrix_2, MatrixSymbol):
            _value_check(scale_matrix_2.is_positive_definite, 'The shape matrix must be positive definite.')
        _value_check(scale_matrix_1.is_square, 'Scale matrix 1 should be be square matrix')
        _value_check(scale_matrix_2.is_square, 'Scale matrix 2 should be be square matrix')
        n = location_matrix.shape[0]
        p = location_matrix.shape[1]
        _value_check(scale_matrix_1.shape[0] == n, 'Scale matrix 1 should be of shape %s x %s' % (str(n), str(n)))
        _value_check(scale_matrix_2.shape[0] == p, 'Scale matrix 2 should be of shape %s x %s' % (str(p), str(p)))

    @property
    def set(self):
        n, p = self.location_matrix.shape
        return MatrixSet(n, p, S.Reals)

    @property
    def dimension(self):
        return self.location_matrix.shape

    def pdf(self, x):
        M, U, V = (self.location_matrix, self.scale_matrix_1, self.scale_matrix_2)
        n, p = M.shape
        if isinstance(x, list):
            x = ImmutableMatrix(x)
        if not isinstance(x, (MatrixBase, MatrixSymbol)):
            raise ValueError('%s should be an isinstance of Matrix or MatrixSymbol' % str(x))
        term1 = Inverse(V) * Transpose(x - M) * Inverse(U) * (x - M)
        num = exp(-Trace(term1) / S(2))
        den = (2 * pi) ** (S(n * p) / 2) * Determinant(U) ** (S(p) / 2) * Determinant(V) ** (S(n) / 2)
        return num / den