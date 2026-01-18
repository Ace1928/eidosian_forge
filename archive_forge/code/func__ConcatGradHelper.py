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
def _ConcatGradHelper(op, grad, start_value_index, end_value_index, dim_index):
    """Gradient for concat op.

  Args:
    op: An operation.
    grad: `Tensor` or `IndexedSlices` representing the gradients with respect to
      each output of the op.
    start_value_index: An integer index of the first value in the op.inputs.
    end_value_index: An integer index of the last value in the op.inputs.
    dim_index: An integer index of concat_dim or axis parameter in op.inputs.

  Returns:
    Tensors representing the partial gradients with respect to each input
    of the op.

  Raises:
    ValueError: if concat_dim/axis is not statically known.
  """

    def _CreateDenseMaskAndBegin(sizes, concat_dim):
        """Create variables for iteratively slicing a dense gradients tensor."""
        shape_of_shape = array_ops.shape(sizes[0])
        mask = array_ops.concat([array_ops.zeros(array_ops.expand_dims(concat_dim, 0), dtype=dtypes.int32), [1], array_ops.zeros(shape_of_shape - concat_dim - 1, dtype=dtypes.int32)], 0)
        begin = array_ops.zeros(shape_of_shape, dtype=dtypes.int32)
        return (mask, begin)

    def _ExtractInputShapes(inputs):
        """Extract the shapes of a set of input tensors."""
        if context.executing_eagerly():
            return array_ops.shape_n(inputs)
        sizes = []
        fully_known = True
        for x in inputs:
            input_shape = array_ops.shape(x)
            if not isinstance(input_shape, tensor.Tensor) or input_shape.op.type != 'Const':
                fully_known = False
                break
            sizes.append(input_shape)
        if fully_known:
            return sizes
        else:
            return array_ops.shape_n(inputs)
    if len(op.inputs) == 2:
        return grad + [None] if end_value_index <= dim_index else [None] + grad
    concat_dim = op.inputs[dim_index]
    input_values = op.inputs[start_value_index:end_value_index]
    out_grads = []
    if isinstance(grad, tensor.Tensor):
        if context.executing_eagerly() or isinstance(concat_dim, ops.EagerTensor):
            non_neg_concat_dim = concat_dim._numpy().item(0) % input_values[0]._rank()
            sizes = pywrap_tfe.TFE_Py_TensorShapeSlice(input_values, non_neg_concat_dim)
            out_grads = array_ops.split(grad, sizes, non_neg_concat_dim)
        else:
            if constant_op.is_constant(concat_dim):
                grad_context = control_flow_util.GetOutputContext(grad.op)
                dim_context = control_flow_util.GetOutputContext(concat_dim.op)
                if dim_context != grad_context:
                    value = tensor_util.constant_value(concat_dim)
                    concat_dim = constant_op.constant(value=value, dtype=concat_dim.dtype)
            non_neg_concat_dim = concat_dim % array_ops.rank(input_values[0])
            sizes = _ExtractInputShapes(input_values)
            if len(sizes) > 16:
                sizes = array_ops.squeeze(array_ops.slice(array_ops_stack.stack(sizes, axis=1), [non_neg_concat_dim, 0], [1, -1]))
                out_grads = array_ops.split(grad, sizes, non_neg_concat_dim)
            else:
                offset = gen_array_ops.concat_offset(non_neg_concat_dim, sizes)
                for begin, size in zip(offset, sizes):
                    out_grads.append(array_ops.slice(grad, begin, size))
    elif isinstance(grad, indexed_slices_lib.IndexedSlices):
        non_neg_concat_dim = concat_dim % array_ops.rank(input_values[0])
        concat_dim_static = tensor_util.constant_value(concat_dim)
        if concat_dim_static is None:
            raise ValueError('Can only compute IndexedSlices gradient with statically-known concat_dim')
        if concat_dim_static < 0:
            rank = tensor_util.constant_value(array_ops.rank(input_values[0]))
            if rank is None:
                raise ValueError('Can only compute IndexedSlices gradient with negative concat_dim when first value rank is statically-known.')
            concat_dim_static %= rank
        sizes = [array_ops.shape(x) for x in input_values]
        if concat_dim_static > 0:
            mask, begin = _CreateDenseMaskAndBegin(sizes, non_neg_concat_dim)
            for size in sizes:
                new_values = array_ops.slice(grad.values, begin, array_ops.concat([[-1], array_ops.slice(size, [1], [-1])], 0))
                out_grads.append(indexed_slices_lib.IndexedSlices(new_values, grad.indices, size))
                begin = math_ops.add(begin, size * mask)
        else:
            start = constant_op.constant(0, dtype=grad.indices.dtype)
            for size in sizes:
                size_concat_dim = array_ops.gather(size, non_neg_concat_dim)
                if size_concat_dim.dtype != grad.indices.dtype:
                    size_concat_dim = math_ops.cast(size_concat_dim, dtype=grad.indices.dtype)
                end = start + size_concat_dim
                indices_to_select = array_ops.squeeze(array_ops.where(math_ops.logical_and(grad.indices >= start, grad.indices < end)), axis=[1])
                new_indices = array_ops.gather(grad.indices, indices_to_select) - start
                new_values = array_ops.gather(grad.values, indices_to_select)
                out_grads.append(indexed_slices_lib.IndexedSlices(new_values, new_indices, size))
                start = end
    else:
        raise TypeError('Expected Tensor or IndexedSlices, got %s' % type(grad))
    return out_grads + [None] if end_value_index <= dim_index else [None] + out_grads