import threading
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import device
from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_nccl_ops
def _apply_all_reduce(reduction, tensors):
    """Helper function for all_* functions."""
    if not tensors:
        raise ValueError('Must pass >0 tensors to all reduce operations')
    shared_name = _get_shared_name()

    def _all_reduce():
        """Call nccl allreduce."""
        res = []
        for t in tensors:
            _check_device(t)
            with ops.device(t.device):
                res.append(gen_nccl_ops.nccl_all_reduce(input=t, reduction=reduction, num_devices=len(tensors), shared_name=shared_name))
        return res
    if context.executing_eagerly():
        return def_function.function(_all_reduce)()
    else:
        return _all_reduce()