import collections
import enum
from typing import Any, Callable, Iterable, List, Optional, Text, Tuple, Union
from absl import logging
import numpy as np
from tensorflow.compiler.tf2xla.python import xla as tf2xla
from tensorflow.core.framework import attr_value_pb2
from tensorflow.core.protobuf.tpu import dynamic_padding_pb2 as dynamic_padding
from tensorflow.core.protobuf.tpu import tpu_embedding_configuration_pb2 as embedding_pb2
from tensorflow.python import tf2
from tensorflow.python.compiler.xla import xla
from tensorflow.python.framework import auto_control_deps
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import config
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import func_graph
from tensorflow.python.framework import function
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import cond
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.tpu import device_assignment as device_assignment_lib
from tensorflow.python.tpu import tensor_tracer
from tensorflow.python.tpu import tpu_feed
from tensorflow.python.tpu import tpu_function
from tensorflow.python.tpu import tpu_name_util
from tensorflow.python.tpu import tpu_replication
from tensorflow.python.tpu.ops import tpu_ops
from tensorflow.python.types import core as core_types
from tensorflow.python.util import compat
from tensorflow.python.util import nest
from tensorflow.python.util import object_identity
from tensorflow.python.util import traceback_utils
from tensorflow.python.util import variable_utils
from tensorflow.python.util.tf_export import tf_export
def _flatten_and_filter_composite(maybe_composite, non_composite_output, composite_output=None):
    """For an input, replaced the input by a tuple if the input is composite.

  If `maybe_composite` is not composite, return the parameter
  `non_composite_output` otherwise return a tuple which consists of the value of
  the parameter `composite_output` the same number of times as there are
  components of the composite tensor.

  This is useful for computing a mask when flattening nested data with
  `expand_composites=True`. For example

  ```python
  nest.flatten(data, expand_composites=True)
  ```

  and

  ```python
  nest.flatten(nest.map(
      data, lambda x: _flatten_and_filter_composite(x, False, True)))
  ```

  will have the same length and second will be True if the tensor in the first
  is derived from a expanding a composite tensor.

  Args:
    maybe_composite: A value to test for being a composite tensor.
    non_composite_output: The value to return when `maybe_composite` is not a
      composite.
    composite_output: the value to fill the output tuple with if
      `maybe_composite` is a composite.

  Returns:
    `non_composite_output` or a tuple with multiple copies of
    `composite_output`.
  """
    if isinstance(maybe_composite, composite_tensor.CompositeTensor):
        num_components = len(nest.flatten(maybe_composite, expand_composites=True))
        return (composite_output,) * num_components
    return non_composite_output