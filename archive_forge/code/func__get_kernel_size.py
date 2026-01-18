import warnings
import cupy
from cupy._core import internal
from cupyx.scipy.ndimage import _util
from cupyx.scipy.ndimage import _filters
from cupyx.scipy.signal import _signaltools_core as _st_core
def _get_kernel_size(kernel_size, ndim):
    if kernel_size is None:
        kernel_size = (3,) * ndim
    kernel_size = _util._fix_sequence_arg(kernel_size, ndim, 'kernel_size', int)
    if any((k % 2 != 1 for k in kernel_size)):
        raise ValueError('Each element of kernel_size should be odd')
    return kernel_size