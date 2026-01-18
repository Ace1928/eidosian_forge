from sympy.core.expr import Expr
from sympy.core.numbers import I
from sympy.core.singleton import S
from sympy.matrices.matrices import MatrixBase
from sympy.matrices import eye, zeros
from sympy.external import import_module
def _numpy_zeros(m, n, **options):
    """numpy version of zeros."""
    dtype = options.get('dtype', 'float64')
    if not np:
        raise ImportError
    return np.zeros((m, n), dtype=dtype)