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
class NegativeMultinomialDistribution(JointDistribution):
    _argnames = ('k0', 'p')
    is_Continuous = False
    is_Discrete = True

    @staticmethod
    def check(k0, p):
        _value_check(k0 > 0, 'number of failures must be a positive integer')
        for p_k in p:
            _value_check((p_k >= 0, p_k <= 1), 'probability must be in range [0, 1].')
        _value_check(sum(p) <= 1, 'success probabilities must not be greater than 1.')

    @property
    def set(self):
        return Range(0, S.Infinity) ** len(self.p)

    def pdf(self, *k):
        k0, p = (self.k0, self.p)
        term_1 = gamma(k0 + sum(k)) * (1 - sum(p)) ** k0 / gamma(k0)
        term_2 = Mul.fromiter((pi ** ki / factorial(ki) for pi, ki in zip(p, k)))
        return term_1 * term_2