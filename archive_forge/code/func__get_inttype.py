import warnings
import numpy
import cupy
def _get_inttype(input):
    nbytes = sum(((x - 1) * abs(stride) for x, stride in zip(input.shape, input.strides))) + input.dtype.itemsize
    return 'int' if nbytes < 1 << 31 else 'ptrdiff_t'