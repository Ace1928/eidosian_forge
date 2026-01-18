from sympy.concrete.products import Product
from sympy.concrete.summations import Sum
from sympy.core.add import Add
from sympy.core.function import Lambda
from sympy.core.mul import Mul
from sympy.core.numbers import (Integer, Rational, pi)
from sympy.core.power import Pow
from sympy.core.relational import Eq
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.core.sympify import sympify
from sympy.functions.combinatorial.factorials import (rf, factorial)
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.special.bessel import besselk
from sympy.functions.special.gamma_functions import gamma
from sympy.matrices.dense import (Matrix, ones)
from sympy.sets.fancysets import Range
from sympy.sets.sets import (Intersection, Interval)
from sympy.tensor.indexed import (Indexed, IndexedBase)
from sympy.matrices import ImmutableMatrix, MatrixSymbol
from sympy.matrices.expressions.determinant import det
from sympy.matrices.expressions.matexpr import MatrixElement
from sympy.stats.joint_rv import JointDistribution, JointPSpace, MarginalDistribution
from sympy.stats.rv import _value_check, random_symbols
class MultivariateNormalDistribution(JointDistribution):
    _argnames = ('mu', 'sigma')
    is_Continuous = True

    @property
    def set(self):
        k = self.mu.shape[0]
        return S.Reals ** k

    @staticmethod
    def check(mu, sigma):
        _value_check(mu.shape[0] == sigma.shape[0], 'Size of the mean vector and covariance matrix are incorrect.')
        if not isinstance(sigma, MatrixSymbol):
            _value_check(sigma.is_positive_semidefinite, 'The covariance matrix must be positive semi definite. ')

    def pdf(self, *args):
        mu, sigma = (self.mu, self.sigma)
        k = mu.shape[0]
        if len(args) == 1 and args[0].is_Matrix:
            args = args[0]
        else:
            args = ImmutableMatrix(args)
        x = args - mu
        density = S.One / sqrt((2 * pi) ** k * det(sigma)) * exp(Rational(-1, 2) * x.transpose() * (sigma.inv() * x))
        return MatrixElement(density, 0, 0)

    def _marginal_distribution(self, indices, sym):
        sym = ImmutableMatrix([Indexed(sym, i) for i in indices])
        _mu, _sigma = (self.mu, self.sigma)
        k = self.mu.shape[0]
        for i in range(k):
            if i not in indices:
                _mu = _mu.row_del(i)
                _sigma = _sigma.col_del(i)
                _sigma = _sigma.row_del(i)
        return Lambda(tuple(sym), S.One / sqrt((2 * pi) ** len(_mu) * det(_sigma)) * exp(Rational(-1, 2) * (_mu - sym).transpose() * (_sigma.inv() * (_mu - sym)))[0])