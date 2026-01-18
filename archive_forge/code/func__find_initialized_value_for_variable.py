import abc
import enum
import functools
import itertools
import os
from tensorflow.core.framework import variable_pb2
from tensorflow.python import pywrap_tensorflow  # pylint: disable=unused-import
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_conversion_registry
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.trackable import base as trackable
from tensorflow.python.util import _pywrap_utils
from tensorflow.python.util import object_identity
from tensorflow.python.util import tf_should_use
from tensorflow.python.util import traceback_utils
from tensorflow.python.util.deprecation import deprecated
from tensorflow.python.util.deprecation import deprecated_args
from tensorflow.python.util.tf_export import tf_export
def _find_initialized_value_for_variable(variable_op):
    """Find the initialized value for a variable op.

  To do so, lookup the variable op in the variables collection.

  Args:
    variable_op: A variable `Operation`.

  Returns:
    A `Tensor` representing the initialized value for the variable or `None`
    if the initialized value could not be found.
  """
    try:
        var_names = [variable_op.node_def.name, variable_op.node_def.name + ':0']
        for collection_name in (ops.GraphKeys.GLOBAL_VARIABLES, ops.GraphKeys.LOCAL_VARIABLES):
            for var in variable_op.graph.get_collection(collection_name):
                if var.name in var_names:
                    return var.initialized_value()
    except AttributeError:
        return None
    return None