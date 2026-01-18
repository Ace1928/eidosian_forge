from cupy._core import _kernel
from cupy._core import _memory_range
from cupy._manipulation import join
from cupy._sorting import search
def _get_memory_ptrs(x):
    if x.dtype.kind != 'c':
        return _get_memory_ptrs_kernel(x)
    return join.concatenate([_get_memory_ptrs_kernel(x.real), _get_memory_ptrs_kernel(x.imag)])