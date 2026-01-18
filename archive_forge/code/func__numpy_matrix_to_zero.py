from sympy.core.expr import Expr
from sympy.core.numbers import I
from sympy.core.singleton import S
from sympy.matrices.matrices import MatrixBase
from sympy.matrices import eye, zeros
from sympy.external import import_module
def _numpy_matrix_to_zero(e):
    """Convert a numpy zero matrix to the zero scalar."""
    if not np:
        raise ImportError
    test = np.zeros_like(e)
    if np.allclose(e, test):
        return 0.0
    else:
        return e