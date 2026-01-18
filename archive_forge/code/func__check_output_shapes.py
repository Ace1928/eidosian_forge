import functools
from typing import Any, Callable, Dict, Iterable, List, Optional, Text, Tuple, Union
from absl import logging
from tensorflow.core.framework import attr_value_pb2
from tensorflow.core.protobuf.tpu import tpu_embedding_configuration_pb2
from tensorflow.python.distribute import device_util
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import distribute_utils
from tensorflow.python.distribute import sharded_variable
from tensorflow.python.distribute import tpu_strategy
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import device as tf_device
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework.tensor_shape import TensorShape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables as tf_variables
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.saved_model import registration
from tensorflow.python.saved_model import save_context
from tensorflow.python.tpu import tpu
from tensorflow.python.tpu import tpu_embedding_v2_utils
from tensorflow.python.tpu import tpu_replication
from tensorflow.python.tpu.ops import tpu_ops
from tensorflow.python.trackable import autotrackable
from tensorflow.python.trackable import base
from tensorflow.python.types import internal as internal_types
from tensorflow.python.util import compat
from tensorflow.python.util import nest
from tensorflow.python.util import tf_inspect
from tensorflow.python.util.tf_export import tf_export
def _check_output_shapes(self, incoming_output_shapes: List[TensorShape]):
    """Check the incoming output shapes against the output shapes stored."""
    nest.assert_same_structure(self._output_shapes, incoming_output_shapes)
    for (path, _), old_output_shape, incoming_output_shape in zip(nest.flatten_with_joined_string_paths(self._feature_config), self._output_shapes, incoming_output_shapes):
        if old_output_shape and incoming_output_shape:
            if (len(incoming_output_shape) == 1 or len(incoming_output_shape) == 2) and len(old_output_shape) > len(incoming_output_shape):
                continue
            if len(old_output_shape) != len(incoming_output_shape) or not self._is_tensor_shape_match(old_output_shape, incoming_output_shape):
                raise ValueError(f'Inconsistent shape founded for input feature {path}, Output shape is set to be {old_output_shape}, But got incoming output shape {incoming_output_shape}')