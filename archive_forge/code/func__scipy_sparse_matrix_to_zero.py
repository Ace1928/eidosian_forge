from sympy.core.expr import Expr
from sympy.core.numbers import I
from sympy.core.singleton import S
from sympy.matrices.matrices import MatrixBase
from sympy.matrices import eye, zeros
from sympy.external import import_module
def _scipy_sparse_matrix_to_zero(e):
    """Convert a scipy.sparse zero matrix to the zero scalar."""
    if not np:
        raise ImportError
    edense = e.todense()
    test = np.zeros_like(edense)
    if np.allclose(edense, test):
        return 0.0
    else:
        return e