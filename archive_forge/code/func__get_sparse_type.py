import numpy
import warnings
import cupy
from cupy.cuda import nccl
from cupyx.distributed import _store
from cupyx.distributed._comm import _Backend
from cupyx.scipy import sparse
def _get_sparse_type(matrix):
    if sparse.isspmatrix_coo(matrix):
        return 'coo'
    elif sparse.isspmatrix_csr(matrix):
        return 'csr'
    elif sparse.isspmatrix_csc(matrix):
        return 'csc'
    else:
        raise TypeError('NCCL is not supported for this type of sparse matrix')