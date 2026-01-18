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
def CircularSymplecticEnsemble(sym, dim):
    """
    Represents Circular Symplectic Ensembles.

    Examples
    ========

    >>> from sympy.stats import CircularSymplecticEnsemble as CSE
    >>> from sympy.stats import joint_eigen_distribution
    >>> C = CSE('S', 1)
    >>> joint_eigen_distribution(C)
    Lambda(t[1], Product(Abs(exp(I*t[_j]) - exp(I*t[_k]))**4, (_j, _k + 1, 1), (_k, 1, 0))/(2*pi))

    Note
    ====

    As can be seen above in the example, density of CiruclarSymplecticEnsemble
    is not evaluated because the exact definition is based on haar measure of
    unitary group which is not unique.
    """
    sym, dim = (_symbol_converter(sym), _sympify(dim))
    model = CircularSymplecticEnsembleModel(sym, dim)
    rmp = RandomMatrixPSpace(sym, model=model)
    return RandomMatrixSymbol(sym, dim, dim, pspace=rmp)