import dataclasses
import functools
import os
import threading
import types as types_lib
import weakref
from google.protobuf import text_format as _text_format
from google.protobuf.message import DecodeError
from tensorflow.core.framework import attr_value_pb2
from tensorflow.core.function import trace_type
from tensorflow.core.function.capture import capture_container
from tensorflow.core.function.polymorphism import function_cache
from tensorflow.python.distribute.parallel_device import parallel_device
from tensorflow.python.eager import context
from tensorflow.python.eager import lift_to_graph
from tensorflow.python.eager import monitoring
from tensorflow.python.eager.polymorphic_function import attributes as attributes_lib
from tensorflow.python.eager.polymorphic_function import autograph_util
from tensorflow.python.eager.polymorphic_function import compiler_ir
from tensorflow.python.eager.polymorphic_function import eager_function_run
from tensorflow.python.eager.polymorphic_function import function_type_utils
from tensorflow.python.eager.polymorphic_function import tf_method_target
from tensorflow.python.eager.polymorphic_function import tracing_compilation
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import errors
from tensorflow.python.framework import func_graph as func_graph_module
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_spec
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import cond
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import control_flow_util
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.profiler import trace
from tensorflow.python.trackable import base as trackable
from tensorflow.python.types import core
from tensorflow.python.util import deprecation
from tensorflow.python.util import nest
from tensorflow.python.util import object_identity
from tensorflow.python.util import tf_decorator
from tensorflow.python.util import traceback_utils
from tensorflow.python.util.tf_export import tf_export
def _list_all_concrete_functions_for_serialization(self):
    """Returns all concrete functions for serialization.

    Returns:
      A list of instances of `ConcreteFunction`.
    """
    seen_signatures = []
    if self.input_signature is not None:
        seen_signatures.append((self.input_signature, {}))
    else:
        concrete_functions = self._list_all_concrete_functions()
        for concrete_function in concrete_functions:
            signature = concrete_function.structured_input_signature
            flattened = nest.flatten(signature)
            if any((isinstance(arg, func_graph_module.UnknownArgument) for arg in flattened)):
                logging.info('Unsupported signature for serialization: %s.', signature)
                continue
            equal_to_signature = functools.partial(function_type_utils.is_same_structure, signature, check_values=True)
            if not any((equal_to_signature(s) for s in seen_signatures)):
                seen_signatures.append(signature)
    concrete_functions = []
    for args, kwargs in seen_signatures:
        concrete_functions.append(self.get_concrete_function(*args, **kwargs))
    return concrete_functions