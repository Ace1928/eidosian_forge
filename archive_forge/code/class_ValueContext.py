import collections
import contextlib
import copy
import enum  # pylint: disable=g-bad-import-order
import functools
import threading
import weakref
import six
from tensorflow.python import tf2
from tensorflow.python.autograph.core import ag_ctx as autograph_ctx
from tensorflow.python.autograph.impl import api as autograph
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.distribute import collective_util
from tensorflow.python.distribute import device_util
from tensorflow.python.distribute import numpy_dataset
from tensorflow.python.distribute import reduce_util
from tensorflow.python.eager import context as eager_context
from tensorflow.python.eager import def_function
from tensorflow.python.eager import monitoring
from tensorflow.python.eager import tape
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import custom_gradient
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import ref_variable
from tensorflow.python.ops import summary_ops_v2
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variable_v1
from tensorflow.python.platform import tf_logging
from tensorflow.python.trackable import base as trackable
from tensorflow.python.types import distribute as ds_types
from tensorflow.python.util import deprecation
from tensorflow.python.util import nest
from tensorflow.python.util import tf_contextlib
from tensorflow.python.util.deprecation import deprecated
from tensorflow.python.util.tf_export import tf_export
from tensorflow.tools.docs import doc_controls
@tf_export('distribute.experimental.ValueContext', v1=[])
class ValueContext(object):
    """A class wrapping information needed by a distribute function.

  This is a context class that is passed to the `value_fn` in
  `strategy.experimental_distribute_values_from_function` and contains
  information about the compute replicas. The `num_replicas_in_sync` and
  `replica_id` can be used to customize the value on each replica.

  Example usage:

  1.  Directly constructed.

      >>> def value_fn(context):
      ...   return context.replica_id_in_sync_group/context.num_replicas_in_sync
      >>> context = tf.distribute.experimental.ValueContext(
      ...   replica_id_in_sync_group=2, num_replicas_in_sync=4)
      >>> per_replica_value = value_fn(context)
      >>> per_replica_value
      0.5

  2.  Passed in by `experimental_distribute_values_from_function`.  {: value=2}

      >>> strategy = tf.distribute.MirroredStrategy(["GPU:0", "GPU:1"])
      >>> def value_fn(value_context):
      ...   return value_context.num_replicas_in_sync
      >>> distributed_values = (
      ...      strategy.experimental_distribute_values_from_function(
      ...        value_fn))
      >>> local_result = strategy.experimental_local_results(distributed_values)
      >>> local_result
      (2, 2)

  """
    __slots__ = ['_replica_id_in_sync_group', '_num_replicas_in_sync']

    def __init__(self, replica_id_in_sync_group=0, num_replicas_in_sync=1):
        """Initializes a ValueContext object.

    Args:
      replica_id_in_sync_group: the current replica_id, should be an int in
        [0,`num_replicas_in_sync`).
      num_replicas_in_sync: the number of replicas that are in sync.
    """
        self._replica_id_in_sync_group = replica_id_in_sync_group
        self._num_replicas_in_sync = num_replicas_in_sync

    @property
    def num_replicas_in_sync(self):
        """Returns the number of compute replicas in sync."""
        return self._num_replicas_in_sync

    @property
    def replica_id_in_sync_group(self):
        """Returns the replica ID."""
        return self._replica_id_in_sync_group

    def __str__(self):
        return 'tf.distribute.ValueContext(replica id {},  total replicas in sync: {})'.format(self.replica_id_in_sync_group, self.num_replicas_in_sync)