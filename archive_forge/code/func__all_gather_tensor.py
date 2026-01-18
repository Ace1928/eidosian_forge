import atexit
import collections
import contextlib
import copy
import functools
import weakref
from absl import logging
import numpy as np
from tensorflow.python.autograph.core import ag_ctx as autograph_ctx
from tensorflow.python.autograph.impl import api as autograph
from tensorflow.python.compiler.xla.experimental import xla_sharding
from tensorflow.python.distribute import cross_device_ops as cross_device_ops_lib
from tensorflow.python.distribute import device_util
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import distribute_utils
from tensorflow.python.distribute import input_lib
from tensorflow.python.distribute import input_util
from tensorflow.python.distribute import numpy_dataset
from tensorflow.python.distribute import reduce_util
from tensorflow.python.distribute import tpu_replicated_variable
from tensorflow.python.distribute import tpu_util
from tensorflow.python.distribute import tpu_values
from tensorflow.python.distribute import values
from tensorflow.python.distribute.cluster_resolver import tpu_cluster_resolver as tpu_cluster_resolver_lib
from tensorflow.python.distribute.v1 import input_lib as input_lib_v1
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.eager import function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import device as tf_device
from tensorflow.python.framework import device_spec
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variables as variables_lib
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.saved_model import save_context
from tensorflow.python.tpu import device_assignment as device_assignment_lib  # pylint: disable=unused-import
from tensorflow.python.tpu import tpu
from tensorflow.python.tpu import tpu_hardware_feature
from tensorflow.python.tpu import training_loop
from tensorflow.python.tpu.ops import tpu_ops
from tensorflow.python.util import deprecation
from tensorflow.python.util import nest
from tensorflow.python.util import tf_inspect
from tensorflow.python.util.tf_export import tf_export
def _all_gather_tensor(value, axis):
    value = ops.convert_to_tensor(value)
    if value.shape.rank is None:
        value_rank = array_ops.rank(value)
        value_shape = array_ops.shape(value)
    else:
        value_rank = value.shape.rank
        value_shape = value.shape.as_list()
        value_shape_tensor = array_ops.shape(value)
        for i in range(len(value_shape)):
            if value_shape[i] is None:
                value_shape[i] = value_shape_tensor[i]
    axis = _make_axis_nonnegative(axis, value_rank)
    if isinstance(value_rank, int):
        replica_broadcast_shape = [1] * (value_rank + 1)
        replica_broadcast_shape[axis] = self.num_replicas_in_sync
    else:
        replica_broadcast_shape = array_ops.where_v2(math_ops.equal(math_ops.range(value_rank + 1), axis), self.num_replicas_in_sync, 1)
    output_shape = self._compute_all_gather_output_shape(value_shape, value_rank, axis)
    if value.dtype in _DTYPES_SUPPORTED_BY_CROSS_REPLICA_SUM:
        replica_id_mask = array_ops.one_hot(self.replica_id_in_sync_group, self.num_replicas_in_sync)
        replica_id_mask = array_ops.reshape(replica_id_mask, replica_broadcast_shape)
        replica_id_mask = math_ops.cast(replica_id_mask, value.dtype)
        gathered_value = array_ops.expand_dims(value, axis) * replica_id_mask
        gathered_value = self.all_reduce(reduce_util.ReduceOp.SUM, gathered_value)
        return array_ops.reshape(gathered_value, output_shape)
    else:
        inputs = array_ops.expand_dims(value, axis=axis)
        inputs = array_ops.tile(inputs, replica_broadcast_shape)
        unordered_output = tpu_ops.all_to_all(inputs, concat_dimension=axis, split_dimension=axis, split_count=self.num_replicas_in_sync)
        concat_replica_id = array_ops.reshape(self.replica_id_in_sync_group, [1])
        concat_replica_id = array_ops.tile(concat_replica_id, [self.num_replicas_in_sync])
        xla_to_replica_context_id = tpu_ops.all_to_all(concat_replica_id, concat_dimension=0, split_dimension=0, split_count=self.num_replicas_in_sync)
        replica_context_to_xla_id = math_ops.argmax(array_ops.one_hot(xla_to_replica_context_id, self.num_replicas_in_sync), axis=0)
        sorted_with_extra_dim = array_ops.gather(unordered_output, replica_context_to_xla_id, axis=axis)
        return array_ops.reshape(sorted_with_extra_dim, output_shape)