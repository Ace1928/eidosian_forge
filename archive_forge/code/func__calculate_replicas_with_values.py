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
def _calculate_replicas_with_values(strategy, input_workers, optional_list):
    """Calcualates the number of replicas that have values.

  Args:
    strategy: the `tf.distribute.Strategy`.
    input_workers: the `InputWorkers`.
    optional_list: a list of lists `tf.experimental.Optional`. The values from
      each compute device grouped by the input device.

  Returns:
    A scalar Tensor.
  """
    worker_has_values = []
    for worker, optionals in zip(input_workers.worker_devices, optional_list):
        with ops.device(worker):
            device_has_values = [math_ops.cast(v.has_value(), dtypes.int64) for v in optionals]
            worker_has_values.append(math_ops.reduce_sum(device_has_values, keepdims=True))
    client_has_values = math_ops.reduce_sum(worker_has_values, keepdims=True)
    if strategy.extended._in_multi_worker_mode():
        global_has_values = strategy.reduce(reduce_util.ReduceOp.SUM, client_has_values, axis=None)
        return array_ops.reshape(global_has_values, [])
    else:
        return array_ops.reshape(client_has_values, [])