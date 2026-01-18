import cupy
import cupyx
from cupyx.scipy import sparse
def _check_A_type(A):
    if not (isinstance(A, cupy.ndarray) or cupyx.scipy.sparse.isspmatrix(A)):
        msg = 'A must be cupy.ndarray or cupyx.scipy.sparse.spmatrix'
        raise TypeError(msg)