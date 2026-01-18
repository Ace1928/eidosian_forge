import functools
import sys
import time
import six
from tensorflow.python.autograph.operators import py_builtins
from tensorflow.python.data.experimental.ops import batching
from tensorflow.python.data.experimental.ops import cardinality as cardinality_lib
from tensorflow.python.data.experimental.ops import distribute
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import iterator_ops
from tensorflow.python.data.ops import multi_device_iterator_ops
from tensorflow.python.data.ops import optional_ops
from tensorflow.python.distribute import device_util
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import distribute_utils
from tensorflow.python.distribute import input_ops
from tensorflow.python.distribute import reduce_util
from tensorflow.python.distribute import values
from tensorflow.python.distribute.distribute_lib import InputReplicationMode
from tensorflow.python.eager import context
from tensorflow.python.eager import monitoring
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import device as tf_device
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework import type_spec
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import cond as tf_cond
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import while_loop
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.types import distribute as distribute_types
from tensorflow.python.util import nest
from tensorflow.python.util.compat import collections_abc
def _create_per_replica(value_list, strategy):
    """Creates a PerReplica.

  For strategies other than OneDeviceStrategy, it creates a PerReplica whose
  type spec is set to the element spec of the dataset. This helps avoid
  retracing for partial batches. Retracing is problematic for multi client when
  different client retraces different time, since retracing changes the
  collective keys in the tf.function, and causes mismatches among clients.

  For single client strategies, this simply calls distribute_utils.regroup().

  Args:
    value_list: a list of values, one for each replica.
    strategy: the `tf.distribute.Strategy`.

  Returns:
    a structure of PerReplica.

  """
    always_wrap = _always_wrap(strategy)
    per_replicas = distribute_utils.regroup(value_list, always_wrap=always_wrap)
    return per_replicas