import abc
from tensorflow.core.framework import attr_value_pb2
from tensorflow.core.protobuf import control_flow_pb2
from tensorflow.python.eager import context
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework import type_spec
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_util as util
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import gen_control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops.gen_control_flow_ops import *
from tensorflow.python.util import compat
from tensorflow.python.util import dispatch
from tensorflow.python.util import nest
from tensorflow.python.util import variable_utils
from tensorflow.python.util.tf_export import tf_export
def _Enter(tensor, frame_name, is_constant=False, parallel_iterations=10, use_ref=True, use_input_shape=True, name=None):
    """Creates or finds a child frame, and makes `tensor` available to it.

  The unique `frame_name` is used by the `Executor` to identify frames. If
  `is_constant` is true, `tensor` is a constant in the child frame; otherwise
  it may be changed in the child frame. At most `parallel_iterations`
  iterations are run in parallel in the child frame.

  Args:
    tensor: The tensor to be made available to the child frame.
    frame_name: The name of the child frame.
    is_constant: If true, the output is constant within the child frame.
    parallel_iterations: The number of iterations allowed to run in parallel.
    use_ref: If true, use ref_enter if tensor is of ref type.
    use_input_shape: If true, set the result's shape based on tensor's shape.
    name: A name for this operation (optional).

  Returns:
    The same tensor as `tensor`.

  Raises:
    ValueError: If any tensor in `tensor` has a less specific shape
      than its corresponding shape in `shape_invariant`.
  """
    tensor = ops.internal_convert_to_tensor_or_composite(tensor, as_ref=True)
    if isinstance(tensor, tensor_lib.Tensor):
        if tensor.dtype._is_ref_dtype and use_ref:
            result = gen_control_flow_ops.ref_enter(tensor, frame_name, is_constant, parallel_iterations, name=name)
        else:
            result = gen_control_flow_ops.enter(tensor, frame_name, is_constant, parallel_iterations, name=name)
        if use_input_shape:
            result.set_shape(tensor.get_shape())
        return result
    elif isinstance(tensor, composite_tensor.CompositeTensor):

        def enter_component(t):
            return _Enter(t, frame_name, is_constant, parallel_iterations, use_ref, use_input_shape)
        return nest.map_structure(enter_component, tensor, expand_composites=True)
    else:
        raise TypeError(f"'tensor' must be a Tensor or CompositeTensor. Received: {type(tensor)}.")