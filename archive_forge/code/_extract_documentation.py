import cupy
import cupyx
from cupyx.scipy import sparse
Returns the upper triangular portion of a matrix in sparse format

    Args:
        A (cupy.ndarray or cupyx.scipy.sparse.spmatrix): Matrix whose upper
            triangular portion is desired.
        k (integer): The bottom-most diagonal of the upper triangle.
        format (string): Sparse format of the result, e.g. 'csr', 'csc', etc.

    Returns:
        cupyx.scipy.sparse.spmatrix:
            Upper triangular portion of A in sparse format.

    .. seealso:: :func:`scipy.sparse.triu`
    