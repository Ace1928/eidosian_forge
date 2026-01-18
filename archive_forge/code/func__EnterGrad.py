from tensorflow.python.framework import dtypes
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import control_flow_util
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.control_flow_ops import *
@ops.RegisterGradient('Enter')
def _EnterGrad(op, grad):
    """Gradients for an Enter are calculated using an Exit op.

  For loop variables, grad is the gradient so just add an exit.
  For loop invariants, we need to add an accumulator loop.
  """
    graph = ops.get_default_graph()
    grad_ctxt = graph._get_control_flow_context()
    if grad_ctxt is None:
        return grad
    if not grad_ctxt.back_prop:
        return grad
    if grad_ctxt.grad_state is None:
        return grad
    if op.get_attr('is_constant'):
        if isinstance(grad, tensor.Tensor):
            result = grad_ctxt.AddBackpropAccumulator(op, grad)
        elif isinstance(grad, indexed_slices.IndexedSlices):
            result = grad_ctxt.AddBackpropIndexedSlicesAccumulator(op, grad)
        else:
            raise TypeError(f'Type {type(grad)} not supported,must be Tensor or Indexed Slices')
    else:
        result = exit(grad)
        grad_ctxt.loop_exits.append(result)
        grad_ctxt.ExitResult([result])
    return result