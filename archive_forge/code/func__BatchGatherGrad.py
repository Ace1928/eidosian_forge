from tensorflow.compiler.tf2xla.ops import gen_xla_ops
from tensorflow.python import pywrap_tfe
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import indexed_slices as indexed_slices_lib
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import cond
from tensorflow.python.ops import control_flow_util
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import gen_resource_variable_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import sparse_ops
def _BatchGatherGrad(params_shape, values, indices, batch_dims, gather_dim_size):
    """Returns the gradient of GatherV2 with batch dimensions."""
    indices_size = array_ops.expand_dims(array_ops.size(indices), 0)
    if batch_dims:
        values_shape = array_ops.shape(values)
        outer_shape = values_shape[:batch_dims]
        inner_shape = values_shape[batch_dims:][1:]
        batch_size = gen_math_ops.prod(outer_shape, [0], False)
        flat_values_shape = array_ops.concat([[-1], inner_shape], 0)
        gather_dim_size *= batch_size
        indices = _GetBatchIndices(params_shape, indices, batch_dims)
        values = array_ops.reshape(_IndexedSlicesToTensorNoWarning(values), flat_values_shape)
    indices = array_ops.reshape(indices, indices_size)
    params_grad = math_ops.unsorted_segment_sum(values, indices, gather_dim_size)
    if batch_dims:
        params_grad = array_ops.reshape(params_grad, array_ops.concat([outer_shape, flat_values_shape], 0))
    return params_grad