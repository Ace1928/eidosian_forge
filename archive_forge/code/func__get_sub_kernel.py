import cupy
from cupy_backends.cuda.api import runtime
from cupy import _util
from cupyx.scipy.ndimage import _filters_core
def _get_sub_kernel(f):
    """
    Takes the "function" given to generic_filter and returns the "sub-kernel"
    that will be called, one of RawKernel or ReductionKernel.

    This supports:
     * cupy.RawKernel
       no checks are possible
     * cupy.ReductionKernel
       checks that there is a single input and output
    """
    if isinstance(f, cupy.RawKernel):
        return f
    elif isinstance(f, cupy.ReductionKernel):
        if f.nin != 1 or f.nout != 1:
            raise TypeError('ReductionKernel must have 1 input and output')
        return f
    elif isinstance(f, cupy.ElementwiseKernel):
        raise TypeError('only ReductionKernel allowed (not ElementwiseKernel)')
    else:
        raise TypeError('bad function type')