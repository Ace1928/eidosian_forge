from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import tensor_array_ops
@ops.RegisterGradient('TensorArrayScatter')
@ops.RegisterGradient('TensorArrayScatterV2')
@ops.RegisterGradient('TensorArrayScatterV3')
def _TensorArrayScatterGrad(op, flow):
    """Gradient for TensorArrayScatter.

  Args:
    op: Forward TensorArrayScatter op.
    flow: Gradient `Tensor` flow to TensorArrayScatter.

  Returns:
    A grad `Tensor`, the gradient created in upstream ReadGrads or PackGrad.
  """
    handle = op.inputs[0]
    indices = op.inputs[1]
    dtype = op.get_attr('T')
    grad_source = _GetGradSource(flow)
    flow_out = array_ops.identity(op.outputs[0], 'flow_out')
    with ops.control_dependencies([flow_out]):
        flow = array_ops.identity(flow, 'write_barrier')
    g = tensor_array_ops.TensorArray(dtype=dtype, handle=handle, flow=flow, colocate_with_first_write_call=False).grad(source=grad_source, flow=flow)
    grad = g.gather(indices)
    return [None, None, grad, flow]