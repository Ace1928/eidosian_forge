from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_sparse_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import sparse_ops
def _SparseDenseCwiseMulOrDivGrad(op, grad, is_mul):
    """Common code for SparseDenseCwise{Mul,Div} gradients."""
    x_indices = op.inputs[0]
    x_shape = op.inputs[2]
    y = op.inputs[3]
    y_shape = math_ops.cast(array_ops.shape(y), dtypes.int64)
    num_added_dims = array_ops.expand_dims(array_ops.size(x_shape) - array_ops.size(y_shape), 0)
    augmented_y_shape = array_ops.concat([array_ops.ones(num_added_dims, ops.dtypes.int64), y_shape], 0)
    scaling = x_shape // augmented_y_shape
    scaled_indices = x_indices // scaling
    scaled_indices = array_ops.slice(scaled_indices, array_ops.concat([[0], num_added_dims], 0), [-1, -1])
    dense_vals = array_ops.gather_nd(y, scaled_indices)
    if is_mul:
        dx = grad * dense_vals
        dy_val = grad * op.inputs[1]
    else:
        dx = grad / dense_vals
        dy_val = grad * (-op.inputs[1] / math_ops.square(dense_vals))
    dy = sparse_ops.sparse_add(array_ops.zeros_like(y), sparse_tensor.SparseTensor(scaled_indices, dy_val, y_shape))
    return (None, dx, None, dy)