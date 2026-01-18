from sympy.core.expr import Expr
from sympy.core.numbers import I
from sympy.core.singleton import S
from sympy.matrices.matrices import MatrixBase
from sympy.matrices import eye, zeros
from sympy.external import import_module
def _sympy_tensor_product(*matrices):
    """Compute the kronecker product of a sequence of SymPy Matrices.
    """
    from sympy.matrices.expressions.kronecker import matrix_kronecker_product
    return matrix_kronecker_product(*matrices)