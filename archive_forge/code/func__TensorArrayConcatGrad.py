from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import tensor_array_ops
@ops.RegisterGradient('TensorArrayConcat')
@ops.RegisterGradient('TensorArrayConcatV2')
@ops.RegisterGradient('TensorArrayConcatV3')
def _TensorArrayConcatGrad(op, grad, unused_lengths_grad):
    """Gradient for TensorArrayConcat.

  Args:
    op: Forward TensorArrayConcat op.
    grad: Gradient `Tensor` to TensorArrayConcat.

  Returns:
    A flow `Tensor`, which can be used in control dependencies to
    force the write of `grad` to the gradient `TensorArray`.
  """
    handle = op.inputs[0]
    flow = op.inputs[1]
    lengths = op.outputs[1]
    dtype = op.get_attr('dtype')
    grad_source = _GetGradSource(grad)
    g = tensor_array_ops.TensorArray(dtype=dtype, handle=handle, flow=flow, colocate_with_first_write_call=False).grad(source=grad_source, flow=flow)
    u_g = g.split(grad, lengths=lengths)
    return [None, u_g.flow]