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
def _get_output_shapes_from_input_shapes(self, input_shapes: List[TensorShape]) -> List[TensorShape]:
    """Get output shapes from the flattened input shapes list."""
    output_shapes = []
    for input_shape, feature in zip(input_shapes, nest.flatten(self._feature_config)):
        if input_shape.rank is None or input_shape.rank < 1:
            raise ValueError('Received input tensor of shape {}. Rank must be 1 and above'.format(input_shape))
        if len(input_shape) == 2 and input_shape[-1] != 1 and (not feature.output_shape) and (feature.max_sequence_length > 0):
            input_shape_list = input_shape.as_list()
            input_shape_list.insert(len(input_shape_list) - 1, feature.max_sequence_length)
            input_shape = TensorShape(input_shape_list)
        if input_shape.rank == 1:
            output_shapes.append(input_shape)
        else:
            output_shapes.append(input_shape[:-1])
    return output_shapes