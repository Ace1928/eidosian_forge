from sympy.concrete.products import Product
from sympy.concrete.summations import Sum
from sympy.core.basic import Basic
from sympy.core.function import Lambda
from sympy.core.numbers import (I, pi)
from sympy.core.singleton import S
from sympy.core.symbol import Dummy
from sympy.functions.elementary.complexes import Abs
from sympy.functions.elementary.exponential import exp
from sympy.functions.special.gamma_functions import gamma
from sympy.integrals.integrals import Integral
from sympy.matrices.expressions.matexpr import MatrixSymbol
from sympy.matrices.expressions.trace import Trace
from sympy.tensor.indexed import IndexedBase
from sympy.core.sympify import _sympify
from sympy.stats.rv import _symbol_converter, Density, RandomMatrixSymbol, is_random
from sympy.stats.joint_rv_types import JointDistributionHandmade
from sympy.stats.random_matrix import RandomMatrixPSpace
from sympy.tensor.array import ArrayComprehension
class GaussianOrthogonalEnsembleModel(GaussianEnsembleModel):

    @property
    def normalization_constant(self):
        n = self.dimension
        _H = MatrixSymbol('_H', n, n)
        return Integral(exp(-S(n) / 4 * Trace(_H ** 2)))

    def density(self, expr):
        n, ZGOE = (self.dimension, self.normalization_constant)
        h_pspace = RandomMatrixPSpace('P', model=self)
        H = RandomMatrixSymbol('H', n, n, pspace=h_pspace)
        return Lambda(H, exp(-S(n) / 4 * Trace(H ** 2)) / ZGOE)(expr)

    def joint_eigen_distribution(self):
        return self._compute_joint_eigen_distribution(S.One)

    def level_spacing_distribution(self):
        s = Dummy('s')
        f = pi / 2 * s * exp(-pi / 4 * s ** 2)
        return Lambda(s, f)