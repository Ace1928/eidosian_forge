import functools
import itertools
import operator
from tensorflow.python.eager import backprop
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import gen_nn_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
@ops.RegisterGradient('ApproxTopK')
def _ApproxTopKGradient(op, grad, _):
    """Return the gradients for ApproxTopK.

  Args:
    op: The ApproxTopK for which we need to generate gradients.
    grad: The gradients for backprop.

  Returns:
    Scattered gradient based on the top-k indices.
  """
    idx_shape = op.outputs[1].shape
    lifted_idx_shape = idx_shape + [1]
    flat_shape_len = functools.reduce(operator.mul, idx_shape)
    rank = idx_shape.rank
    reduction_dim = op.get_attr('reduction_dimension')
    if reduction_dim < 0:
        reduction_dim = rank + reduction_dim

    def GetLiftedIdx(d):
        if d == reduction_dim:
            return array_ops.reshape(op.outputs[1], lifted_idx_shape)
        iota_len = idx_shape[d]
        iota_shape = list(itertools.repeat(1, rank + 1))
        iota_shape[d] = iota_len
        iota = array_ops.reshape(math_ops.range(iota_len), iota_shape)
        return array_ops.broadcast_to(iota, lifted_idx_shape)
    lifted_idx = array_ops.concat(list((GetLiftedIdx(d) for d in range(rank))), axis=rank)
    flat_idx = array_ops.reshape(lifted_idx, [flat_shape_len, rank])
    flat_grad = array_ops.reshape(grad, [flat_shape_len])
    return array_ops.scatter_nd(flat_idx, flat_grad, op.inputs[0].shape)