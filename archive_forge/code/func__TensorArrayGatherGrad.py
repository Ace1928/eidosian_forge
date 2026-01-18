from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import tensor_array_ops
@ops.RegisterGradient('TensorArrayGather')
@ops.RegisterGradient('TensorArrayGatherV2')
@ops.RegisterGradient('TensorArrayGatherV3')
def _TensorArrayGatherGrad(op, grad):
    """Gradient for TensorArrayGather.

  Args:
    op: Forward TensorArrayGather op.
    grad: Gradient `Tensor` to TensorArrayGather.

  Returns:
    A flow `Tensor`, which can be used in control dependencies to
    force the write of `grad` to the gradient `TensorArray`.
  """
    handle = op.inputs[0]
    indices = op.inputs[1]
    flow = op.inputs[2]
    dtype = op.get_attr('dtype')
    grad_source = _GetGradSource(grad)
    g = tensor_array_ops.TensorArray(dtype=dtype, handle=handle, flow=flow, colocate_with_first_write_call=False).grad(source=grad_source, flow=flow)
    u_g = g.scatter(indices, grad)
    return [None, None, u_g.flow]