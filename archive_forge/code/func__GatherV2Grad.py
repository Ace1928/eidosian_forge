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
@ops.RegisterGradient('GatherV2')
def _GatherV2Grad(op, grad):
    """Gradient for GatherV2 op."""
    params = op.inputs[0]
    with ops.colocate_with(params):
        params_shape = array_ops.shape(params, out_type=ops.dtypes.int64)
        params_shape = math_ops.cast(params_shape, dtypes.int32)
    indices = op.inputs[1]
    indices_size = array_ops.expand_dims(array_ops.size(indices), 0)
    axis = op.inputs[2]
    axis_static = tensor_util.constant_value(axis)
    batch_dims = int(op.get_attr('batch_dims'))
    if batch_dims < 0:
        if indices.shape.ndims is None:
            raise ValueError(f'Currently, it is unsupported to take the gradient of tf.gather when batch_dims < 0 and the rank of the indices is unknown. Please pass a positive batch_dims or use tf.ensure_shape to update the shape of indices when calling tf.gather. Got batch_dims={batch_dims} and indices={indices}')
        batch_dims += indices.shape.ndims
    if axis_static == 0:
        if context.executing_eagerly():
            with ops.device(indices_size.device):
                params_tail_shape = array_ops.identity(params_shape)[1:]
        else:
            params_tail_shape = params_shape[1:]
        values_shape = array_ops.concat([indices_size, params_tail_shape], 0)
        values = array_ops.reshape(_IndexedSlicesToTensorNoWarning(grad), values_shape)
        indices = array_ops.reshape(indices, indices_size)
        params_grad = indexed_slices_lib.IndexedSlices(values, indices, params_shape)
    else:
        outer_shape = params_shape[:axis]
        inner_shape = params_shape[axis:][1:]
        values_shape = array_ops.concat([outer_shape, [-1], inner_shape], 0)
        values_dims = array_ops.size(values_shape)
        axis_dims = array_ops.size(outer_shape)
        outer_batches_indices = math_ops.range(batch_dims)
        batch_axis_indices = math_ops.range(batch_dims, axis_dims)
        inner_axes_indices = math_ops.range(axis_dims + 1, values_dims)
        values = array_ops.reshape(_IndexedSlicesToTensorNoWarning(grad), values_shape)
        transpose_dims = array_ops.concat([outer_batches_indices, [axis_dims], batch_axis_indices, inner_axes_indices], 0)
        values_transpose = array_ops.transpose(values, transpose_dims)
        params_shape_transpose = array_ops.gather(params_shape, transpose_dims)
        params_grad = _BatchGatherGrad(params_shape_transpose, values_transpose, indices, batch_dims, params_shape[axis])
        invert_transpose_dims = array_ops.concat([outer_batches_indices, batch_axis_indices + 1, [batch_dims], inner_axes_indices], 0)
        params_grad = array_ops.transpose(params_grad, invert_transpose_dims)
    return [params_grad, None, None]