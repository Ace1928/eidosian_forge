import functools as _functools
import numpy as _numpy
import platform as _platform
import cupy as _cupy
from cupy_backends.cuda.api import driver as _driver
from cupy_backends.cuda.api import runtime as _runtime
from cupy_backends.cuda.libs import cusparse as _cusparse
from cupy._core import _dtype
from cupy.cuda import device as _device
from cupy.cuda import stream as _stream
from cupy import _util
import cupyx.scipy.sparse
class SpVecDescriptor(BaseDescriptor):

    @classmethod
    def create(cls, idx, x):
        nnz = x.size
        cuda_dtype = _dtype.to_cuda_dtype(x.dtype)
        desc = _cusparse.createSpVec(nnz, nnz, idx.data.ptr, x.data.ptr, _dtype_to_IndexType(idx.dtype), _cusparse.CUSPARSE_INDEX_BASE_ZERO, cuda_dtype)
        get = _cusparse.spVecGet
        destroy = _cusparse.destroySpVec
        return SpVecDescriptor(desc, get, destroy)