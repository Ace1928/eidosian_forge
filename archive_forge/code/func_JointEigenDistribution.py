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
def JointEigenDistribution(mat):
    """
    Creates joint distribution of eigen values of matrices with random
    expressions.

    Parameters
    ==========

    mat: Matrix
        The matrix under consideration.

    Returns
    =======

    JointDistributionHandmade

    Examples
    ========

    >>> from sympy.stats import Normal, JointEigenDistribution
    >>> from sympy import Matrix
    >>> A = [[Normal('A00', 0, 1), Normal('A01', 0, 1)],
    ... [Normal('A10', 0, 1), Normal('A11', 0, 1)]]
    >>> JointEigenDistribution(Matrix(A))
    JointDistributionHandmade(-sqrt(A00**2 - 2*A00*A11 + 4*A01*A10 + A11**2)/2
    + A00/2 + A11/2, sqrt(A00**2 - 2*A00*A11 + 4*A01*A10 + A11**2)/2 + A00/2 + A11/2)

    """
    eigenvals = mat.eigenvals(multiple=True)
    if not all((is_random(eigenval) for eigenval in set(eigenvals))):
        raise ValueError('Eigen values do not have any random expression, joint distribution cannot be generated.')
    return JointDistributionHandmade(*eigenvals)