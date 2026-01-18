import warnings
import numpy
from numpy import linalg
import cupy
from cupy._core import internal
from cupy.cuda import device
from cupy.linalg import _decomposition
from cupy.linalg import _util
from cupy.cublas import batched_gesv, get_batched_gesv_limit
import cupyx
def _nrm2_last_axis(x):
    real_dtype = x.dtype.char.lower()
    x = cupy.ascontiguousarray(x)
    return cupy.sum(cupy.square(x.view(real_dtype)), axis=-1)