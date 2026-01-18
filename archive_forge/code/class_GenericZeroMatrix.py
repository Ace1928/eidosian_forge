from sympy.assumptions.ask import ask, Q
from sympy.core.relational import Eq
from sympy.core.singleton import S
from sympy.core.sympify import _sympify
from sympy.functions.special.tensor_functions import KroneckerDelta
from sympy.matrices.common import NonInvertibleMatrixError
from .matexpr import MatrixExpr
class GenericZeroMatrix(ZeroMatrix):
    """
    A zero matrix without a specified shape

    This exists primarily so MatAdd() with no arguments can return something
    meaningful.
    """

    def __new__(cls):
        return super(ZeroMatrix, cls).__new__(cls)

    @property
    def rows(self):
        raise TypeError('GenericZeroMatrix does not have a specified shape')

    @property
    def cols(self):
        raise TypeError('GenericZeroMatrix does not have a specified shape')

    @property
    def shape(self):
        raise TypeError('GenericZeroMatrix does not have a specified shape')

    def __eq__(self, other):
        return isinstance(other, GenericZeroMatrix)

    def __ne__(self, other):
        return not self == other

    def __hash__(self):
        return super().__hash__()