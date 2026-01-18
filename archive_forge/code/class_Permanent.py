from sympy.core.basic import Basic
from sympy.core.expr import Expr
from sympy.core.singleton import S
from sympy.core.sympify import sympify
from sympy.matrices.common import NonSquareMatrixError
from sympy.assumptions.ask import ask, Q
from sympy.assumptions.refine import handlers_dict
class Permanent(Expr):
    """Matrix Permanent

    Represents the permanent of a matrix expression.

    Examples
    ========

    >>> from sympy import MatrixSymbol, Permanent, ones
    >>> A = MatrixSymbol('A', 3, 3)
    >>> Permanent(A)
    Permanent(A)
    >>> Permanent(ones(3, 3)).doit()
    6
    """

    def __new__(cls, mat):
        mat = sympify(mat)
        if not mat.is_Matrix:
            raise TypeError('Input to Permanent, %s, not a matrix' % str(mat))
        return Basic.__new__(cls, mat)

    @property
    def arg(self):
        return self.args[0]

    def doit(self, expand=False, **hints):
        try:
            return self.arg.per()
        except (AttributeError, NotImplementedError):
            return self