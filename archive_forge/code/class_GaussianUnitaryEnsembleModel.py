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
class GaussianUnitaryEnsembleModel(GaussianEnsembleModel):

    @property
    def normalization_constant(self):
        n = self.dimension
        return 2 ** (S(n) / 2) * pi ** (S(n ** 2) / 2)

    def density(self, expr):
        n, ZGUE = (self.dimension, self.normalization_constant)
        h_pspace = RandomMatrixPSpace('P', model=self)
        H = RandomMatrixSymbol('H', n, n, pspace=h_pspace)
        return Lambda(H, exp(-S(n) / 2 * Trace(H ** 2)) / ZGUE)(expr)

    def joint_eigen_distribution(self):
        return self._compute_joint_eigen_distribution(S(2))

    def level_spacing_distribution(self):
        s = Dummy('s')
        f = 32 / pi ** 2 * s ** 2 * exp(-4 / pi * s ** 2)
        return Lambda(s, f)