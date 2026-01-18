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
class MatrixStudentTDistribution(MatrixDistribution):
    _argnames = ('nu', 'location_matrix', 'scale_matrix_1', 'scale_matrix_2')

    @staticmethod
    def check(nu, location_matrix, scale_matrix_1, scale_matrix_2):
        if not isinstance(scale_matrix_1, MatrixSymbol):
            _value_check(scale_matrix_1.is_positive_definite != False, 'The shape matrix must be positive definite.')
        if not isinstance(scale_matrix_2, MatrixSymbol):
            _value_check(scale_matrix_2.is_positive_definite != False, 'The shape matrix must be positive definite.')
        _value_check(scale_matrix_1.is_square != False, 'Scale matrix 1 should be be square matrix')
        _value_check(scale_matrix_2.is_square != False, 'Scale matrix 2 should be be square matrix')
        n = location_matrix.shape[0]
        p = location_matrix.shape[1]
        _value_check(scale_matrix_1.shape[0] == p, 'Scale matrix 1 should be of shape %s x %s' % (str(p), str(p)))
        _value_check(scale_matrix_2.shape[0] == n, 'Scale matrix 2 should be of shape %s x %s' % (str(n), str(n)))
        _value_check(nu.is_positive != False, 'Degrees of freedom must be positive')

    @property
    def set(self):
        n, p = self.location_matrix.shape
        return MatrixSet(n, p, S.Reals)

    @property
    def dimension(self):
        return self.location_matrix.shape

    def pdf(self, x):
        from sympy.matrices.dense import eye
        if isinstance(x, list):
            x = ImmutableMatrix(x)
        if not isinstance(x, (MatrixBase, MatrixSymbol)):
            raise ValueError('%s should be an isinstance of Matrix or MatrixSymbol' % str(x))
        nu, M, Omega, Sigma = (self.nu, self.location_matrix, self.scale_matrix_1, self.scale_matrix_2)
        n, p = M.shape
        K = multigamma((nu + n + p - 1) / 2, p) * Determinant(Omega) ** (-n / 2) * Determinant(Sigma) ** (-p / 2) / (pi ** (n * p / 2) * multigamma((nu + p - 1) / 2, p))
        return K * Determinant(eye(n) + Inverse(Sigma) * (x - M) * Inverse(Omega) * Transpose(x - M)) ** (-(nu + n + p - 1) / 2)