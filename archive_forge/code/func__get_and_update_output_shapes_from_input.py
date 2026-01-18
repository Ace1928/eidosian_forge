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
def _get_and_update_output_shapes_from_input(self, per_replica_input_shapes: Optional[List[TensorShape]]=None, per_replica_batch_size: Optional[int]=None):
    """Get and update the per replica output shapes from the input."""
    per_replica_output_shapes = None
    if per_replica_batch_size and per_replica_input_shapes is None:
        logging.warning('per_replica_batch_size argument will be deprecated, please specify all the input shapes using per_replica_input_shapes argument.')
        per_replica_output_shapes = self._get_output_shapes_from_batch_size(per_replica_batch_size)
    if per_replica_input_shapes is not None:
        if isinstance(per_replica_input_shapes, int):
            logging.warning('Passing batch size to per_replica_input_shapes argument will be deprecated, please specify all the input shapes using per_replica_input_shapes argument.')
            per_replica_output_shapes = self._get_output_shapes_from_batch_size(per_replica_input_shapes)
        else:
            nest.assert_same_structure(nest.flatten(per_replica_input_shapes), nest.flatten(self._feature_config))
            per_replica_input_shapes = nest.flatten(per_replica_input_shapes)
            per_replica_output_shapes = self._get_output_shapes_from_input_shapes(per_replica_input_shapes)
    if per_replica_output_shapes is not None:
        self._check_output_shapes(per_replica_output_shapes)
        self._update_output_shapes(per_replica_output_shapes)
    self._check_output_shapes_fully_defined()