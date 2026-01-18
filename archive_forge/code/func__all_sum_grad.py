import threading
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import device
from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_nccl_ops
@ops.RegisterGradient('NcclAllReduce')
def _all_sum_grad(op, grad):
    """The gradients for `all_sum`.

  Args:
    op: The `all_sum` `Operation` that we are differentiating.
    grad: Gradient with respect to the output of the `all_sum` op.

  Returns:
    The gradient with respect to the output of `all_sum`.

  Raises:
    LookupError: If `reduction` is not `sum`.
  """
    if op.get_attr('reduction') != b'sum':
        raise LookupError('No gradient defined for NcclAllReduce except for reduction="sum".')
    _check_device(grad, expected=op.device)
    num_devices = op.get_attr('num_devices')
    shared_name = op.get_attr('shared_name') + b'_grad'
    with ops.device(op.device):
        return gen_nccl_ops.nccl_all_reduce(input=grad, reduction='sum', num_devices=num_devices, shared_name=shared_name)