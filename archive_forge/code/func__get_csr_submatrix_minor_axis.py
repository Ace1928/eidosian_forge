import cupy
from cupy import _core
from cupyx.scipy.sparse._base import isspmatrix
from cupyx.scipy.sparse._base import spmatrix
from cupy_backends.cuda.libs import cusparse
from cupy.cuda import device
from cupy.cuda import runtime
import numpy
def _get_csr_submatrix_minor_axis(Ax, Aj, Ap, start, stop):
    """Return a submatrix of the input sparse matrix by slicing minor axis.

    Args:
        Ax (cupy.ndarray): data array from input sparse matrix
        Aj (cupy.ndarray): indices array from input sparse matrix
        Ap (cupy.ndarray): indptr array from input sparse matrix
        start (int): starting index of minor axis
        stop (int): ending index of minor axis

    Returns:
        Bx (cupy.ndarray): data array of output sparse matrix
        Bj (cupy.ndarray): indices array of output sparse matrix
        Bp (cupy.ndarray): indptr array of output sparse matrix

    """
    mask = (start <= Aj) & (Aj < stop)
    mask_sum = cupy.empty(Aj.size + 1, dtype=Aj.dtype)
    mask_sum[0] = 0
    mask_sum[1:] = mask
    cupy.cumsum(mask_sum, out=mask_sum)
    Bp = mask_sum[Ap]
    Bj = Aj[mask] - start
    Bx = Ax[mask]
    return (Bx, Bj, Bp)