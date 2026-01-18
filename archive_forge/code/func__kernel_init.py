import warnings
import numpy
import cupy
from cupy import _core
from cupy import _util
def _kernel_init():
    return _core.ElementwiseKernel('X x', 'Y y', 'if (x == 0) { y = -1; } else { y = i; }', 'cupyx_scipy_ndimage_label_init')