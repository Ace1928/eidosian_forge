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
class RestoredFunction(def_function.Function):
    """Wrapper class for a function that has been restored from saved state.

  See `def_function.Function`.
  """

    def __init__(self, python_function, name, function_spec, concrete_functions):
        super(RestoredFunction, self).__init__(python_function, name, autograph=False, jit_compile=function_spec.jit_compile)
        self.concrete_functions = concrete_functions
        self._function_type = function_spec.function_type
        self._default_values = function_spec.default_values
        self._omit_frequent_tracing_warning = True

    @property
    def _run_functions_eagerly(self):
        return False

    def _list_all_concrete_functions(self):
        return self.concrete_functions

    def _list_all_concrete_functions_for_serialization(self):
        return self.concrete_functions