import collections
import pprint
from tensorflow.core.framework import attr_value_pb2
from tensorflow.core.function.polymorphism import function_type as function_type_lib
from tensorflow.python import pywrap_tfe
from tensorflow.python.eager import backprop_util
from tensorflow.python.eager import context
from tensorflow.python.eager import forwardprop_util
from tensorflow.python.eager import record
from tensorflow.python.eager.graph_only_ops import graph_placeholder
from tensorflow.python.eager.polymorphic_function import atomic_function
from tensorflow.python.eager.polymorphic_function import attributes as attributes_lib
from tensorflow.python.eager.polymorphic_function import function_type_utils
from tensorflow.python.eager.polymorphic_function import saved_model_exported_concrete
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import func_graph as func_graph_module
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import type_spec
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import default_gradient
from tensorflow.python.ops import gradients_util
from tensorflow.python.ops import handle_data_util
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.profiler import trace
from tensorflow.python.trackable import base as trackable
from tensorflow.python.types import core
from tensorflow.python.util import _pywrap_utils
from tensorflow.python.util import compat
from tensorflow.python.util import nest
from tensorflow.python.util import object_identity
def _structured_signature_summary(self, default_values=False):
    """Returns a string summarizing this function's structured signature.

    Args:
      default_values: If true, then include default values in the signature.

    Returns:
      A `string`.
    """
    assert self.function_type is not None
    arg_specs, kwarg_specs = self.structured_input_signature
    arg_names = function_type_utils.to_arg_names(self.function_type)
    arg_names = arg_names[:len(arg_specs)]
    if default_values:
        for i in range(len(arg_names)):
            if not _contains_type_spec(arg_specs[i]):
                arg_names[i] += '={}'.format(arg_specs[i])
    if kwarg_specs:
        arg_names.append('*')
        for name, spec in kwarg_specs.items():
            arg_names.append(name)
            if default_values and (not _contains_type_spec(spec)):
                arg_names[-1] += '={}'.format(spec)
    signature = f'{self._func_graph.name}({', '.join(arg_names)})'
    return signature