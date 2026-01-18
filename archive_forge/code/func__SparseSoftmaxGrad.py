from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_sparse_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import sparse_ops
@ops.RegisterGradient('SparseSoftmax')
def _SparseSoftmaxGrad(op, grad):
    """Gradients for SparseSoftmax.

  The calculation is the same as SoftmaxGrad:

    grad_x = grad_softmax * softmax - sum(grad_softmax * softmax) * softmax

  where we now only operate on the non-zero values present in the SparseTensors.

  Args:
    op: the SparseSoftmax op.
    grad: the upstream gradient w.r.t. the non-zero SparseSoftmax output values.

  Returns:
    Gradients w.r.t. the input (sp_indices, sp_values, sp_shape).
  """
    indices, shape = (op.inputs[0], op.inputs[2])
    out_vals = op.outputs[0]
    sp_output = sparse_tensor.SparseTensor(indices, out_vals, shape)
    sp_grad = sparse_tensor.SparseTensor(indices, grad, shape)
    sp_product = sparse_tensor.SparseTensor(indices, sp_output.values * sp_grad.values, shape)
    sum_reduced = -sparse_ops.sparse_reduce_sum(sp_product, [-1], keepdims=True)
    sp_sum = sparse_ops.sparse_dense_cwise_add(sp_grad, sum_reduced)
    grad_x = sp_sum.values * sp_output.values
    return [None, grad_x, None]