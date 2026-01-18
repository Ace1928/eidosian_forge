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
class GaussianEnsembleModel(RandomMatrixEnsembleModel):
    """
    Abstract class for Gaussian ensembles.
    Contains the properties common to all the
    gaussian ensembles.

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Random_matrix#Gaussian_ensembles
    .. [2] https://arxiv.org/pdf/1712.07903.pdf
    """

    def _compute_normalization_constant(self, beta, n):
        """
        Helper function for computing normalization
        constant for joint probability density of eigen
        values of Gaussian ensembles.

        References
        ==========

        .. [1] https://en.wikipedia.org/wiki/Selberg_integral#Mehta's_integral
        """
        n = S(n)
        prod_term = lambda j: gamma(1 + beta * S(j) / 2) / gamma(S.One + beta / S(2))
        j = Dummy('j', integer=True, positive=True)
        term1 = Product(prod_term(j), (j, 1, n)).doit()
        term2 = (2 / (beta * n)) ** (beta * n * (n - 1) / 4 + n / 2)
        term3 = (2 * pi) ** (n / 2)
        return term1 * term2 * term3

    def _compute_joint_eigen_distribution(self, beta):
        """
        Helper function for computing the joint
        probability distribution of eigen values
        of the random matrix.
        """
        n = self.dimension
        Zbn = self._compute_normalization_constant(beta, n)
        l = IndexedBase('l')
        i = Dummy('i', integer=True, positive=True)
        j = Dummy('j', integer=True, positive=True)
        k = Dummy('k', integer=True, positive=True)
        term1 = exp(-S(n) / 2 * Sum(l[k] ** 2, (k, 1, n)).doit())
        sub_term = Lambda(i, Product(Abs(l[j] - l[i]) ** beta, (j, i + 1, n)))
        term2 = Product(sub_term(i).doit(), (i, 1, n - 1)).doit()
        syms = ArrayComprehension(l[k], (k, 1, n)).doit()
        return Lambda(tuple(syms), term1 * term2 / Zbn)