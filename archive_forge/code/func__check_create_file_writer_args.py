import abc
import collections
import functools
import os
import re
import threading
from tensorflow.core.framework import graph_pb2
from tensorflow.core.framework import summary_pb2
from tensorflow.core.protobuf import config_pb2
from tensorflow.python.eager import context
from tensorflow.python.eager import profiler as _profiler
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import smart_cond
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_resource_variable_ops
from tensorflow.python.ops import gen_summary_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import summary_op_util
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.trackable import resource
from tensorflow.python.training import training_util
from tensorflow.python.util import deprecation
from tensorflow.python.util import tf_contextlib
from tensorflow.python.util.lazy_loader import LazyLoader
from tensorflow.python.util.tf_export import tf_export
def _check_create_file_writer_args(inside_function, **kwargs):
    """Helper to check the validity of arguments to a create_file_writer() call.

  Args:
    inside_function: whether the create_file_writer() call is in a tf.function
    **kwargs: the arguments to check, as kwargs to give them names.

  Raises:
    ValueError: if the arguments are graph tensors.
  """
    for arg_name, arg in kwargs.items():
        if not isinstance(arg, ops.EagerTensor) and tensor_util.is_tf_type(arg):
            if inside_function:
                raise ValueError(f"Invalid graph Tensor argument '{arg_name}={arg}' to create_file_writer() inside an @tf.function. The create call will be lifted into the outer eager execution context, so it cannot consume graph tensors defined inside the function body.")
            else:
                raise ValueError(f"Invalid graph Tensor argument '{arg_name}={arg}' to eagerly executed create_file_writer().")