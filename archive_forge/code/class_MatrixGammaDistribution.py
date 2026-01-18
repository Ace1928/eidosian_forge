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
class MatrixGammaDistribution(MatrixDistribution):
    _argnames = ('alpha', 'beta', 'scale_matrix')

    @staticmethod
    def check(alpha, beta, scale_matrix):
        if not isinstance(scale_matrix, MatrixSymbol):
            _value_check(scale_matrix.is_positive_definite, 'The shape matrix must be positive definite.')
        _value_check(scale_matrix.is_square, 'Should be square matrix')
        _value_check(alpha.is_positive, 'Shape parameter should be positive.')
        _value_check(beta.is_positive, 'Scale parameter should be positive.')

    @property
    def set(self):
        k = self.scale_matrix.shape[0]
        return MatrixSet(k, k, S.Reals)

    @property
    def dimension(self):
        return self.scale_matrix.shape

    def pdf(self, x):
        alpha, beta, scale_matrix = (self.alpha, self.beta, self.scale_matrix)
        p = scale_matrix.shape[0]
        if isinstance(x, list):
            x = ImmutableMatrix(x)
        if not isinstance(x, (MatrixBase, MatrixSymbol)):
            raise ValueError('%s should be an isinstance of Matrix or MatrixSymbol' % str(x))
        sigma_inv_x = -Inverse(scale_matrix) * x / beta
        term1 = exp(Trace(sigma_inv_x)) / (beta ** (p * alpha) * multigamma(alpha, p))
        term2 = Determinant(scale_matrix) ** (-alpha)
        term3 = Determinant(x) ** (alpha - S(p + 1) / 2)
        return term1 * term2 * term3