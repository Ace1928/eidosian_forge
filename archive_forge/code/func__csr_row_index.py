import cupy
from cupy import _core
from cupyx.scipy.sparse._base import isspmatrix
from cupyx.scipy.sparse._base import spmatrix
from cupy_backends.cuda.libs import cusparse
from cupy.cuda import device
from cupy.cuda import runtime
import numpy
def _csr_row_index(Ax, Aj, Ap, rows):
    """Populate indices and data arrays from the given row index
    Args:
        Ax (cupy.ndarray): data array from input sparse matrix
        Aj (cupy.ndarray): indices array from input sparse matrix
        Ap (cupy.ndarray): indptr array from input sparse matrix
        rows (cupy.ndarray): index array of rows to populate
    Returns:
        Bx (cupy.ndarray): data array of output sparse matrix
        Bj (cupy.ndarray): indices array of output sparse matrix
        Bp (cupy.ndarray): indptr array for output sparse matrix
    """
    row_nnz = cupy.diff(Ap)
    Bp = cupy.empty(rows.size + 1, dtype=Ap.dtype)
    Bp[0] = 0
    cupy.cumsum(row_nnz[rows], out=Bp[1:])
    nnz = int(Bp[-1])
    out_rows = _csr_indptr_to_coo_rows(nnz, Bp)
    Bj, Bx = _csr_row_index_ker(out_rows, rows, Ap, Aj, Ax, Bp)
    return (Bx, Bj, Bp)