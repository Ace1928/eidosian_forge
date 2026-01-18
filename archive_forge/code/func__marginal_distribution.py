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
def _marginal_distribution(self, indices, *sym):
    if len(indices) == 2:
        return self.pdf(*sym)
    if indices[0] == 0:
        x = sym[0]
        v, mu, sigma = (self.alpha - S.Half, self.mu, S(self.beta) / (self.lamda * self.alpha))
        return Lambda(sym, gamma((v + 1) / 2) / (gamma(v / 2) * sqrt(pi * v) * sigma) * (1 + 1 / v * ((x - mu) / sigma) ** 2) ** ((-v - 1) / 2))
    from sympy.stats.crv_types import GammaDistribution
    return Lambda(sym, GammaDistribution(self.alpha, self.beta)(sym[0]))