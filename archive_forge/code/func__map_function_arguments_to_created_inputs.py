import collections
import os
import re
import sys
import traceback
from typing import Any, Callable, Dict, List, Tuple
from absl import logging
from tensorflow.core.framework import function_pb2
from tensorflow.core.framework import graph_pb2
from tensorflow.core.framework import versions_pb2
from tensorflow.core.protobuf import meta_graph_pb2
from tensorflow.core.protobuf import saved_model_pb2
from tensorflow.core.protobuf import saved_object_graph_pb2
from tensorflow.python.checkpoint import checkpoint
from tensorflow.python.checkpoint import checkpoint_options
from tensorflow.python.checkpoint import functional_saver
from tensorflow.python.checkpoint import graph_view
from tensorflow.python.checkpoint import save_util_v1
from tensorflow.python.checkpoint import util as checkpoint_util
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.eager import function as defun
from tensorflow.python.eager.polymorphic_function import concrete_function as cf
from tensorflow.python.eager.polymorphic_function import polymorphic_function
from tensorflow.python.eager.polymorphic_function import saved_model_exported_concrete
from tensorflow.python.eager.polymorphic_function import saved_model_utils
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import function as framework_fn
from tensorflow.python.framework import meta_graph
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework import versions
from tensorflow.python.lib.io import file_io
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.saved_model import builder_impl
from tensorflow.python.saved_model import fingerprinting_utils
from tensorflow.python.saved_model import function_serialization
from tensorflow.python.saved_model import path_helpers
from tensorflow.python.saved_model import pywrap_saved_model
from tensorflow.python.saved_model import registration
from tensorflow.python.saved_model import revived_types
from tensorflow.python.saved_model import save_context
from tensorflow.python.saved_model import save_options
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import signature_def_utils
from tensorflow.python.saved_model import signature_serialization
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model import tracing_utils
from tensorflow.python.saved_model import utils_impl
from tensorflow.python.saved_model.pywrap_saved_model import constants
from tensorflow.python.saved_model.pywrap_saved_model import metrics
from tensorflow.python.trackable import asset
from tensorflow.python.trackable import base
from tensorflow.python.trackable import resource
from tensorflow.python.trackable import trackable_utils
from tensorflow.python.training.saving import trace_saveable_util
from tensorflow.python.types import core as types_core
from tensorflow.python.util import compat
from tensorflow.python.util import object_identity
from tensorflow.python.util import tf_stack
from tensorflow.python.util.tf_export import tf_export
def _map_function_arguments_to_created_inputs(function_arguments: List[Any], signature_key: str, function_name: bytes, defaults=None):
    """Creates exterior placeholders in the exported graph for function arguments.

  Functions have two types of inputs: tensors captured from the outside (eager)
  context, and arguments to the function which we expect to receive from the
  user at each call. `_map_captures_to_created_tensors` replaces
  captured tensors with stand-ins (typically these are resource dtype tensors
  associated with variables). `_map_function_inputs_to_created_inputs` runs over
  every argument, creating a new placeholder for each which will belong to the
  exported graph rather than the function body.

  Args:
    function_arguments: A list of argument placeholders in the function body.
    signature_key: The name of the signature being exported, for error messages.
    function_name: The name of the function, for error messages.
    defaults: A dictionary mapping signature_key to dictionary of
      user_specified_name to Tensor representing default values.

  Returns:
    A tuple of (mapped_inputs, exterior_placeholders)
      mapped_inputs: A list with entries corresponding to `function_arguments`
        containing all of the inputs of the function gathered from the exported
        graph (both captured resources and arguments).
      exterior_argument_placeholders: A dictionary mapping from argument names
        to placeholders in the exported graph, containing the explicit arguments
        to the function which a user is expected to provide.

  Raises:
    ValueError: If argument names are not unique.
  """
    exterior_argument_placeholders = {}
    mapped_inputs = []
    for placeholder in function_arguments:
        user_input_name = compat.as_str_any(placeholder.op.get_attr('_user_specified_name'))
        if user_input_name != placeholder.op.name:
            raise ValueError(f"Got non-flat/non-unique argument names for SavedModel signature '{signature_key}': more than one argument to '{compat.as_str_any(function_name)}' was named '{user_input_name}'. Signatures have one Tensor per named input, so to have predictable names Python functions used to generate these signatures should avoid *args and Tensors in nested structures unless unique names are specified for each. Use tf.TensorSpec(..., name=...) to provide a name for a Tensor input.")
        default_value = defaults.get(signature_key, {}).get(user_input_name)
        if default_value is not None:
            placeholder_with_default = array_ops.placeholder_with_default(input=default_value.numpy(), shape=placeholder.shape, name=_to_safe_name_scope(signature_key, user_input_name))
            exterior_argument_placeholders[user_input_name] = placeholder_with_default
            mapped_inputs.append(placeholder_with_default)
        else:
            arg_placeholder = array_ops.placeholder(shape=placeholder.shape, dtype=placeholder.dtype, name=_to_safe_name_scope(signature_key, user_input_name))
            exterior_argument_placeholders[user_input_name] = arg_placeholder
            mapped_inputs.append(arg_placeholder)
    return (mapped_inputs, exterior_argument_placeholders)