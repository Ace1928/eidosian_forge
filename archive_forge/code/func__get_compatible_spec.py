import collections
from tensorflow.core.framework import types_pb2
from tensorflow.python.eager import backprop_util
from tensorflow.python.framework import auto_control_deps
from tensorflow.python.framework import auto_control_deps_utils as acd
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import func_graph as func_graph_module
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework import type_spec
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_util
from tensorflow.python.ops import control_flow_util_v2 as util
from tensorflow.python.ops import default_gradient
from tensorflow.python.ops import gen_functional_ops
from tensorflow.python.ops import gen_optional_ops
from tensorflow.python.ops import gradients_util
from tensorflow.python.ops import handle_data_util
from tensorflow.python.ops import math_ops
from tensorflow.python.util import nest
def _get_compatible_spec(value_or_spec1, value_or_spec2):
    """Returns the most specific compatible spec.

  Args:
    value_or_spec1: A TypeSpecs or a value that has a defined TypeSpec.
    value_or_spec2: A TypeSpecs or a value that has a defined TypeSpec.

  Returns:
    The most specific compatible TypeSpecs of the input.

  Raises:
    ValueError: If value_or_spec1 is not compatible with value_or_spec2.
  """
    spec1 = _get_spec_for(value_or_spec1)
    spec2 = _get_spec_for(value_or_spec2)
    common = spec1._without_tensor_names().most_specific_common_supertype([spec2._without_tensor_names()])
    if common is None:
        raise TypeError(f'No common supertype of {spec1} and {spec2}.')
    return common