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
def _dtype_to_IndexType(dtype):
    if dtype == 'uint16':
        return _cusparse.CUSPARSE_INDEX_16U
    elif dtype == 'int32':
        return _cusparse.CUSPARSE_INDEX_32I
    elif dtype == 'int64':
        return _cusparse.CUSPARSE_INDEX_64I
    else:
        raise TypeError