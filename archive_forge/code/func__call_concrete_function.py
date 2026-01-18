import collections
import pprint
import re
from absl import logging
from tensorflow.core.function import trace_type
from tensorflow.core.function.polymorphism import function_type as function_type_lib
from tensorflow.core.protobuf import saved_object_graph_pb2
from tensorflow.python.eager import def_function
from tensorflow.python.eager import function as function_lib
from tensorflow.python.eager.polymorphic_function import function_type_utils
from tensorflow.python.framework import func_graph as func_graph_lib
from tensorflow.python.framework import function_def_to_graph as function_def_lib
from tensorflow.python.framework import op_def_registry
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.framework import type_spec
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import custom_gradient
from tensorflow.python.ops import default_gradient
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.saved_model import nested_structure_coder
from tensorflow.python.util import compat
from tensorflow.python.util import nest
from tensorflow.python.util import tf_decorator
from tensorflow.python.util import tf_inspect
def _call_concrete_function(function, inputs):
    """Calls a restored Function with structured inputs.

  This differs from `function.__call__` in that inputs and outputs are
  structured and that it casts inputs to tensors if needed.

  Note: this does not checks that non-tensor inputs match. That should be
  done before via `_concrete_function_callable_with`.

  Args:
    function: ConcreteFunction to call.
    inputs: Structured inputs compatible with
      `function.graph.structured_input_signature`.

  Returns:
    The structured function output.
  """
    expected_structure = function.graph.structured_input_signature
    flatten_inputs = nest.flatten_up_to(expected_structure, inputs, expand_composites=True)
    flatten_expected = nest.flatten(expected_structure, expand_composites=True)
    tensor_inputs = []
    for arg, expected in zip(flatten_inputs, flatten_expected):
        if isinstance(expected, tensor.TensorSpec):
            tensor_inputs.append(ops.convert_to_tensor(arg, dtype_hint=expected.dtype))
        elif isinstance(expected, resource_variable_ops.VariableSpec):
            tensor_inputs.append(arg.handle)
    result = function._call_flat(tensor_inputs, function.captured_inputs)
    if isinstance(result, ops.Operation):
        return None
    return result