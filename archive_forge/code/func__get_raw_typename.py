import warnings
import cupy
from cupy_backends.cuda.api import runtime
import cupyx.scipy.signal._signaltools as filtering
from cupyx.scipy.signal._arraytools import (
from cupyx.scipy.signal.windows._windows import get_window
def _get_raw_typename(dtype):
    return cupy.dtype(dtype).name