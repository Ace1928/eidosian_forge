import warnings
import cupy
from cupy_backends.cuda.api import runtime
import cupyx.scipy.signal._signaltools as filtering
from cupyx.scipy.signal._arraytools import (
from cupyx.scipy.signal.windows._windows import get_window
def _lombscargle(x, y, freqs, pgram, y_dot):
    device_id = cupy.cuda.Device()
    num_blocks = device_id.attributes['MultiProcessorCount'] * 20
    block_sz = 512
    lombscargle_kernel = _get_module_func_raw(LOMBSCARGLE_MODULE, '_cupy_lombscargle', x)
    args = (x.shape[0], freqs.shape[0], x, y, freqs, pgram, y_dot)
    lombscargle_kernel((num_blocks,), (block_sz,), args)