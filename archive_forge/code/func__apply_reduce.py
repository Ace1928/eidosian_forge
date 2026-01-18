import threading
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import device
from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_nccl_ops
def _apply_reduce(reduction, tensors):
    """Helper function for reduce_* functions."""
    if not tensors:
        raise ValueError('Must pass >0 tensors to reduce operations')
    for t in tensors:
        _check_device(t)
    result = gen_nccl_ops.nccl_reduce(input=tensors, reduction=reduction)
    try:
        next((t for t in tensors if t.device == result.device))
    except StopIteration:
        raise ValueError('One input tensor must be assigned to current device')
    return result