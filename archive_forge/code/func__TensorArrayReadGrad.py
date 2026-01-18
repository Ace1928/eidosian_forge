from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import tensor_array_ops
@ops.RegisterGradient('TensorArrayRead')
@ops.RegisterGradient('TensorArrayReadV2')
@ops.RegisterGradient('TensorArrayReadV3')
def _TensorArrayReadGrad(op, grad):
    """Gradient for TensorArrayRead.

  Args:
    op: Forward TensorArrayRead op.
    grad: Gradient `Tensor` to TensorArrayRead.

  Returns:
    A flow `Tensor`, which can be used in control dependencies to
    force the write of `grad` to the gradient `TensorArray`.
  """
    handle = op.inputs[0]
    index = op.inputs[1]
    flow = op.inputs[2]
    dtype = op.get_attr('dtype')
    grad_source = _GetGradSource(grad)
    g = tensor_array_ops.TensorArray(dtype=dtype, handle=handle, flow=flow, colocate_with_first_write_call=False).grad(source=grad_source, flow=flow)
    w_g = g.write(index, grad)
    return [None, None, w_g.flow]