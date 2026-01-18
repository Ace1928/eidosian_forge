from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import func_graph
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import tensor
from tensorflow.python.framework import tensor_conversion
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_resource_variable_ops
from tensorflow.python.util import nest
def _create_grad_indexed_slices_init(grad_output_slices, forward_input):
    """Creates an IndexedSlices to pass as input to the while grad function.

  Args:
    grad_output_slices: IndexedSlices. The corresponding while grad function
      output.
    forward_input: Tensor. The corresponding input to the forward while op.

  Returns:
    Zeros IndexedSlices, created in current Graph.
  """
    assert isinstance(grad_output_slices, indexed_slices.IndexedSlices)
    assert isinstance(forward_input, tensor.Tensor)
    values_out = grad_output_slices.values
    indices_out = grad_output_slices.indices
    if values_out.shape.is_fully_defined():
        values_shape = tensor_shape.TensorShape([0] + values_out.shape.as_list()[1:])
        values = array_ops.zeros(values_shape, dtype=values_out.dtype, name='values_init')
    else:
        if forward_input.dtype == dtypes.resource:
            forward_shape = gen_resource_variable_ops.variable_shape(forward_input)
        else:
            forward_shape = array_ops.shape(forward_input)
        values_shape = array_ops.concat([[0], forward_shape[1:]], 0)
        values = array_ops.zeros(values_shape, dtype=values_out.dtype, name='values_init')
    indices = constant_op.constant([], indices_out.dtype, name='indices_init')
    if forward_input.dtype == dtypes.resource:
        shape = gen_resource_variable_ops.variable_shape(forward_input, name='shape_init')
    else:
        shape = array_ops.shape(forward_input, name='shape_init')
    return indexed_slices.IndexedSlices(values=values, indices=indices, dense_shape=shape)