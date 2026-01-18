import operator
import warnings
import numpy
import cupy
from cupy._core import _accelerator
from cupy.cuda import cub
from cupy.cuda import runtime
from cupyx import cusparse
from cupyx.scipy.sparse import _base
from cupyx.scipy.sparse import _compressed
from cupyx.scipy.sparse import _csc
from cupyx.scipy.sparse import SparseEfficiencyWarning
from cupyx.scipy.sparse import _util
def _maximum_minimum(self, other, cupy_op, op_name, dense_check):
    if _util.isscalarlike(other):
        other = cupy.asarray(other, dtype=self.dtype)
        if dense_check(other):
            dtype = self.dtype
            if dtype == numpy.float32:
                dtype = numpy.float64
            elif dtype == numpy.complex64:
                dtype = numpy.complex128
            dtype = cupy.result_type(dtype, other)
            other = other.astype(dtype, copy=False)
            new_array = cupy_op(self.todense(), other)
            return csr_matrix(new_array)
        else:
            self.sum_duplicates()
            new_data = cupy_op(self.data, other)
            return csr_matrix((new_data, self.indices, self.indptr), shape=self.shape, dtype=self.dtype)
    elif _util.isdense(other):
        self.sum_duplicates()
        other = cupy.atleast_2d(other)
        return cupy_op(self.todense(), other)
    elif isspmatrix_csr(other):
        self.sum_duplicates()
        other.sum_duplicates()
        return binopt_csr(self, other, op_name)
    raise NotImplementedError