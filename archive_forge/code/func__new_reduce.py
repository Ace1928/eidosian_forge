import inspect
import numpy as np
import triton
import triton.language as tl
from .._C.libtriton.triton import interpreter as _interpreter
def _new_reduce(input, axis, combine_fn):
    fn = combine_fn.fn.__name__
    mapping = {'maximum': np.max, '_sum_combine': np.sum}
    ret = mapping[fn](input.handle.data, axis=axis)
    ret_type = tl.block_type(input.dtype, ret.shape)
    return tl.core.tensor(TensorHandle(ret, input.dtype), ret_type)