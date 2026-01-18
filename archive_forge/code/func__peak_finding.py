import math
import cupy
from cupy._core._scalar import get_typename
from cupy_backends.cuda.api import runtime
from cupyx import jit
def _peak_finding(data, comparator, axis, order, mode, results):
    comp = _modedict[comparator]
    clip = mode == 'clip'
    device_id = cupy.cuda.Device()
    num_blocks = (device_id.attributes['MultiProcessorCount'] * 20,)
    block_sz = (512,)
    call_args = (data.shape[axis], order, clip, comp, data, results)
    kernel_name = 'boolrelextrema_1D'
    if data.ndim > 1:
        kernel_name = 'boolrelextrema_2D'
        block_sz_x, block_sz_y = (16, 16)
        n_blocks_x = (data.shape[1] + block_sz_x - 1) // block_sz_x
        n_blocks_y = (data.shape[0] + block_sz_y - 1) // block_sz_y
        block_sz = (block_sz_x, block_sz_y)
        num_blocks = (n_blocks_x, n_blocks_y)
        call_args = (data.shape[1], data.shape[0], order, clip, comp, axis, data, results)
    boolrelextrema = _get_module_func(ARGREL_MODULE, kernel_name, data)
    boolrelextrema(num_blocks, block_sz, call_args)