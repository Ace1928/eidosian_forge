from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_sparse_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import sparse_ops
@ops.RegisterGradient('SparseTensorDenseMatMul')
def _SparseTensorDenseMatMulGrad(op, grad):
    """Gradients for the dense tensor in the SparseTensorDenseMatMul op.

  Args:
    op: the SparseTensorDenseMatMul op
    grad: the incoming gradient

  Returns:
    Gradient for each of the 4 input tensors:
      (sparse_indices, sparse_values, sparse_shape, dense_tensor)
    The gradients for indices and shape are None.

  Raises:
    TypeError: When the two operands don't have the same type.
  """
    a_indices, a_values, a_shape = op.inputs[:3]
    b = op.inputs[3]
    adj_a = op.get_attr('adjoint_a')
    adj_b = op.get_attr('adjoint_b')
    a_type = a_values.dtype.base_dtype
    b_type = b.dtype.base_dtype
    if a_type != b_type:
        raise TypeError(f'SparseTensorDenseMatMul op received operands with different types: `{a_type}` and `{b_type}`.')
    b_grad = gen_sparse_ops.sparse_tensor_dense_mat_mul(a_indices, a_values, a_shape, grad, adjoint_a=not adj_a)
    if adj_b:
        b_grad = array_ops.matrix_transpose(b_grad, conjugate=True)
    rows = a_indices[:, 0]
    cols = a_indices[:, 1]
    parts_a = array_ops.gather(grad, rows if not adj_a else cols)
    parts_b = array_ops.gather(b if not adj_b else array_ops.transpose(b), cols if not adj_a else rows)
    if not adj_a and (not adj_b):
        a_values_grad = math_ops.matmul(array_ops.expand_dims(parts_a, -2), array_ops.expand_dims(parts_b, -2), adjoint_b=True)
    elif adj_a and (not adj_b):
        a_values_grad = math_ops.matmul(array_ops.expand_dims(parts_a, -1), array_ops.expand_dims(parts_b, -1), adjoint_a=True)
    elif not adj_a and adj_b:
        a_values_grad = math_ops.matmul(array_ops.expand_dims(parts_a, -2), array_ops.expand_dims(parts_b, -1))
    elif adj_a and adj_b:
        a_values_grad = math_ops.matmul(array_ops.expand_dims(parts_a, -1), array_ops.expand_dims(parts_b, -2), adjoint_a=True, adjoint_b=True)
    return (None, array_ops.squeeze(a_values_grad, axis=[-2, -1]), None, b_grad)