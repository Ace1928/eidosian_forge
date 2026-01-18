import collections
import contextlib
import os
import re
import threading
import time
import weakref
from six.moves import queue
from tensorflow.python.distribute.coordinator import coordinator_context
from tensorflow.python.distribute.coordinator import metric_utils
from tensorflow.python.distribute.coordinator import remote_value
from tensorflow.python.distribute.coordinator import utils
from tensorflow.python.distribute.coordinator import values as values_lib
from tensorflow.python.distribute.coordinator import watchdog
from tensorflow.python.eager import cancellation
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.eager import executor
from tensorflow.python.eager import function as tf_function
from tensorflow.python.framework import errors
from tensorflow.python.framework import func_graph
from tensorflow.python.framework import ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import tf_export
Blocking call to fetch results from the remote values.

    This is a wrapper around
    `tf.distribute.experimental.coordinator.RemoteValue.fetch` for a
    `RemoteValue` structure; it returns the execution results of
    `RemoteValue`s. If not ready, wait for them while blocking the caller.

    Example:
    ```python
    strategy = ...
    coordinator = tf.distribute.experimental.coordinator.ClusterCoordinator(
        strategy)

    def dataset_fn():
      return tf.data.Dataset.from_tensor_slices([1, 1, 1])

    with strategy.scope():
      v = tf.Variable(initial_value=0)

    @tf.function
    def worker_fn(iterator):
      def replica_fn(x):
        v.assign_add(x)
        return v.read_value()
      return strategy.run(replica_fn, args=(next(iterator),))

    distributed_dataset = coordinator.create_per_worker_dataset(dataset_fn)
    distributed_iterator = iter(distributed_dataset)
    result = coordinator.schedule(worker_fn, args=(distributed_iterator,))
    assert coordinator.fetch(result) == 1
    ```

    Args:
      val: The value to fetch the results from. If this is structure of
        `tf.distribute.experimental.coordinator.RemoteValue`, `fetch()` will be
        called on the individual
        `tf.distribute.experimental.coordinator.RemoteValue` to get the result.

    Returns:
      If `val` is a `tf.distribute.experimental.coordinator.RemoteValue` or a
      structure of `tf.distribute.experimental.coordinator.RemoteValue`s,
      return the fetched `tf.distribute.experimental.coordinator.RemoteValue`
      values immediately if they are available, or block the call until they are
      available, and return the fetched
      `tf.distribute.experimental.coordinator.RemoteValue` values with the same
      structure. If `val` is other types, return it as-is.
    