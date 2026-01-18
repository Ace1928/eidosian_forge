from sympy.assumptions.ask import ask, Q
from sympy.core.relational import Eq
from sympy.core.singleton import S
from sympy.core.sympify import _sympify
from sympy.functions.special.tensor_functions import KroneckerDelta
from sympy.matrices.common import NonInvertibleMatrixError
from .matexpr import MatrixExpr
class GenericIdentity(Identity):
    """
    An identity matrix without a specified shape

    This exists primarily so MatMul() with no arguments can return something
    meaningful.
    """

    def __new__(cls):
        return super(Identity, cls).__new__(cls)

    @property
    def rows(self):
        raise TypeError('GenericIdentity does not have a specified shape')

    @property
    def cols(self):
        raise TypeError('GenericIdentity does not have a specified shape')

    @property
    def shape(self):
        raise TypeError('GenericIdentity does not have a specified shape')

    @property
    def is_square(self):
        return True

    def __eq__(self, other):
        return isinstance(other, GenericIdentity)

    def __ne__(self, other):
        return not self == other

    def __hash__(self):
        return super().__hash__()