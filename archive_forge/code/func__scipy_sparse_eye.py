from sympy.core.expr import Expr
from sympy.core.numbers import I
from sympy.core.singleton import S
from sympy.matrices.matrices import MatrixBase
from sympy.matrices import eye, zeros
from sympy.external import import_module
def _scipy_sparse_eye(n):
    """scipy.sparse version of complex eye."""
    if not sparse:
        raise ImportError
    return sparse.eye(n, n, dtype='complex')