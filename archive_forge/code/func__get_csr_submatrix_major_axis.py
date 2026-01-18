import cupy
from cupy import _core
from cupyx.scipy.sparse._base import isspmatrix
from cupyx.scipy.sparse._base import spmatrix
from cupy_backends.cuda.libs import cusparse
from cupy.cuda import device
from cupy.cuda import runtime
import numpy
def _get_csr_submatrix_major_axis(Ax, Aj, Ap, start, stop):
    """Return a submatrix of the input sparse matrix by slicing major axis.

    Args:
        Ax (cupy.ndarray): data array from input sparse matrix
        Aj (cupy.ndarray): indices array from input sparse matrix
        Ap (cupy.ndarray): indptr array from input sparse matrix
        start (int): starting index of major axis
        stop (int): ending index of major axis

    Returns:
        Bx (cupy.ndarray): data array of output sparse matrix
        Bj (cupy.ndarray): indices array of output sparse matrix
        Bp (cupy.ndarray): indptr array of output sparse matrix

    """
    Ap = Ap[start:stop + 1]
    start_offset, stop_offset = (int(Ap[0]), int(Ap[-1]))
    Bp = Ap - start_offset
    Bj = Aj[start_offset:stop_offset]
    Bx = Ax[start_offset:stop_offset]
    return (Bx, Bj, Bp)