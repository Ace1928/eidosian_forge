from sympy.core.expr import Expr
from sympy.core.numbers import I
from sympy.core.singleton import S
from sympy.matrices.matrices import MatrixBase
from sympy.matrices import eye, zeros
from sympy.external import import_module
def _scipy_sparse_zeros(m, n, **options):
    """scipy.sparse version of zeros."""
    spmatrix = options.get('spmatrix', 'csr')
    dtype = options.get('dtype', 'float64')
    if not sparse:
        raise ImportError
    if spmatrix == 'lil':
        return sparse.lil_matrix((m, n), dtype=dtype)
    elif spmatrix == 'csr':
        return sparse.csr_matrix((m, n), dtype=dtype)