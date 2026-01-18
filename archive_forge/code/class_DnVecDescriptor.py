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
class DnVecDescriptor(BaseDescriptor):

    @classmethod
    def create(cls, x):
        cuda_dtype = _dtype.to_cuda_dtype(x.dtype)
        desc = _cusparse.createDnVec(x.size, x.data.ptr, cuda_dtype)
        get = _cusparse.dnVecGet
        destroy = _cusparse.destroyDnVec
        return DnVecDescriptor(desc, get, destroy)