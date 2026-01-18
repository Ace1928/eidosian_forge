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
class DnMatDescriptor(BaseDescriptor):

    @classmethod
    def create(cls, a):
        assert a.ndim == 2
        assert a.flags.f_contiguous
        rows, cols = a.shape
        ld = rows
        cuda_dtype = _dtype.to_cuda_dtype(a.dtype)
        desc = _cusparse.createDnMat(rows, cols, ld, a.data.ptr, cuda_dtype, _cusparse.CUSPARSE_ORDER_COL)
        get = _cusparse.dnMatGet
        destroy = _cusparse.destroyDnMat
        return DnMatDescriptor(desc, get, destroy)