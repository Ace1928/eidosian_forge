import cupy
from cupy import _core
from cupyx.scipy.sparse._base import isspmatrix
from cupyx.scipy.sparse._base import spmatrix
from cupy_backends.cuda.libs import cusparse
from cupy.cuda import device
from cupy.cuda import runtime
import numpy
def _csr_sample_values(n_row, n_col, Ap, Aj, Ax, Bi, Bj, not_found_val=0):
    """Populate data array for a set of rows and columns
    Args
        n_row : total number of rows in input array
        n_col : total number of columns in input array
        Ap : indptr array for input sparse matrix
        Aj : indices array for input sparse matrix
        Ax : data array for input sparse matrix
        Bi : array of rows to extract from input sparse matrix
        Bj : array of columns to extract from input sparse matrix
    Returns
        Bx : data array for output sparse matrix
    """
    Bi[Bi < 0] += n_row
    Bj[Bj < 0] += n_col
    return _csr_sample_values_kern(n_row, n_col, Ap, Aj, Ax, Bi, Bj, not_found_val, size=Bi.size)