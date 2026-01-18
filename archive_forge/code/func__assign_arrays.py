import numpy
import warnings
import cupy
from cupy.cuda import nccl
from cupyx.distributed import _store
from cupyx.distributed._comm import _Backend
from cupyx.scipy import sparse
def _assign_arrays(matrix, arrays, shape):
    if sparse.isspmatrix_coo(matrix):
        matrix.data = arrays[0]
        matrix.row = arrays[1]
        matrix.col = arrays[2]
        matrix._shape = tuple(shape)
    elif sparse.isspmatrix_csr(matrix) or sparse.isspmatrix_csc(matrix):
        matrix.data = arrays[0]
        matrix.indptr = arrays[1]
        matrix.indices = arrays[2]
        matrix._shape = tuple(shape)
    else:
        raise TypeError('NCCL is not supported for this type of sparse matrix')