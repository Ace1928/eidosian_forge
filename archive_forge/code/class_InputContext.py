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
@tf_export('distribute.InputContext')
class InputContext(object):
    """A class wrapping information needed by an input function.

  This is a context class that is passed to the user's input function and
  contains information about the compute replicas and input pipelines. The
  number of compute replicas (in sync training) helps compute the local batch
  size from the desired global batch size for each replica. The input pipeline
  information can be used to return a different subset of the input in each
  replica (for e.g. shard the input pipeline, use a different input
  source etc).
  """
    __slots__ = ['_num_input_pipelines', '_input_pipeline_id', '_num_replicas_in_sync']

    def __init__(self, num_input_pipelines=1, input_pipeline_id=0, num_replicas_in_sync=1):
        """Initializes an InputContext object.

    Args:
      num_input_pipelines: the number of input pipelines in a cluster.
      input_pipeline_id: the current input pipeline id, should be an int in
        [0,`num_input_pipelines`).
      num_replicas_in_sync: the number of replicas that are in sync.
    """
        self._num_input_pipelines = num_input_pipelines
        self._input_pipeline_id = input_pipeline_id
        self._num_replicas_in_sync = num_replicas_in_sync

    @property
    def num_replicas_in_sync(self):
        """Returns the number of compute replicas in sync."""
        return self._num_replicas_in_sync

    @property
    def input_pipeline_id(self):
        """Returns the input pipeline ID."""
        return self._input_pipeline_id

    @property
    def num_input_pipelines(self):
        """Returns the number of input pipelines."""
        return self._num_input_pipelines

    def get_per_replica_batch_size(self, global_batch_size):
        """Returns the per-replica batch size.

    Args:
      global_batch_size: the global batch size which should be divisible by
        `num_replicas_in_sync`.

    Returns:
      the per-replica batch size.

    Raises:
      ValueError: if `global_batch_size` not divisible by
        `num_replicas_in_sync`.
    """
        if global_batch_size % self._num_replicas_in_sync != 0:
            raise ValueError('The `global_batch_size` %r is not divisible by `num_replicas_in_sync` %r ' % (global_batch_size, self._num_replicas_in_sync))
        return global_batch_size // self._num_replicas_in_sync

    def __str__(self):
        return 'tf.distribute.InputContext(input pipeline id {}, total: {})'.format(self.input_pipeline_id, self.num_input_pipelines)