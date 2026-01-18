import cupy
from cupy import _core
from cupyx.scipy.sparse._base import isspmatrix
from cupyx.scipy.sparse._base import spmatrix
from cupy_backends.cuda.libs import cusparse
from cupy.cuda import device
from cupy.cuda import runtime
import numpy
def _try_is_scipy_spmatrix(index):
    if scipy_available:
        return isinstance(index, scipy.sparse.spmatrix)
    return False