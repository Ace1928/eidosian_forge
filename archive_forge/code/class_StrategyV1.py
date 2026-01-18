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
@tf_export(v1=['distribute.Strategy'])
class StrategyV1(StrategyBase):
    """A list of devices with a state & compute distribution policy.

  See [the guide](https://www.tensorflow.org/guide/distribute_strategy)
  for overview and examples.

  Note: Not all `tf.distribute.Strategy` implementations currently support
  TensorFlow's partitioned variables (where a single variable is split across
  multiple devices) at this time.
  """

    def make_dataset_iterator(self, dataset):
        """Makes an iterator for input provided via `dataset`.

    DEPRECATED: This method is not available in TF 2.x.

    Data from the given dataset will be distributed evenly across all the
    compute replicas. We will assume that the input dataset is batched by the
    global batch size. With this assumption, we will make a best effort to
    divide each batch across all the replicas (one or more workers).
    If this effort fails, an error will be thrown, and the user should instead
    use `make_input_fn_iterator` which provides more control to the user, and
    does not try to divide a batch across replicas.

    The user could also use `make_input_fn_iterator` if they want to
    customize which input is fed to which replica/worker etc.

    Args:
      dataset: `tf.data.Dataset` that will be distributed evenly across all
        replicas.

    Returns:
      An `tf.distribute.InputIterator` which returns inputs for each step of the
      computation.  User should call `initialize` on the returned iterator.
    """
        return self._extended._make_dataset_iterator(dataset)

    def make_input_fn_iterator(self, input_fn, replication_mode=InputReplicationMode.PER_WORKER):
        """Returns an iterator split across replicas created from an input function.

    DEPRECATED: This method is not available in TF 2.x.

    The `input_fn` should take an `tf.distribute.InputContext` object where
    information about batching and input sharding can be accessed:

    ```
    def input_fn(input_context):
      batch_size = input_context.get_per_replica_batch_size(global_batch_size)
      d = tf.data.Dataset.from_tensors([[1.]]).repeat().batch(batch_size)
      return d.shard(input_context.num_input_pipelines,
                     input_context.input_pipeline_id)
    with strategy.scope():
      iterator = strategy.make_input_fn_iterator(input_fn)
      replica_results = strategy.experimental_run(replica_fn, iterator)
    ```

    The `tf.data.Dataset` returned by `input_fn` should have a per-replica
    batch size, which may be computed using
    `input_context.get_per_replica_batch_size`.

    Args:
      input_fn: A function taking a `tf.distribute.InputContext` object and
        returning a `tf.data.Dataset`.
      replication_mode: an enum value of `tf.distribute.InputReplicationMode`.
        Only `PER_WORKER` is supported currently, which means there will be
        a single call to `input_fn` per worker. Replicas will dequeue from the
        local `tf.data.Dataset` on their worker.

    Returns:
      An iterator object that should first be `.initialize()`-ed. It may then
      either be passed to `strategy.experimental_run()` or you can
      `iterator.get_next()` to get the next value to pass to
      `strategy.extended.call_for_each_replica()`.
    """
        return super(StrategyV1, self).make_input_fn_iterator(input_fn, replication_mode)

    def experimental_make_numpy_dataset(self, numpy_input, session=None):
        """Makes a tf.data.Dataset for input provided via a numpy array.

    This avoids adding `numpy_input` as a large constant in the graph,
    and copies the data to the machine or machines that will be processing
    the input.

    Note that you will likely need to use
    tf.distribute.Strategy.experimental_distribute_dataset
    with the returned dataset to further distribute it with the strategy.

    Example:
    ```
    numpy_input = np.ones([10], dtype=np.float32)
    dataset = strategy.experimental_make_numpy_dataset(numpy_input)
    dist_dataset = strategy.experimental_distribute_dataset(dataset)
    ```

    Args:
      numpy_input: A nest of NumPy input arrays that will be converted into a
      dataset. Note that lists of Numpy arrays are stacked, as that is normal
      `tf.data.Dataset` behavior.
      session: (TensorFlow v1.x graph execution only) A session used for
        initialization.

    Returns:
      A `tf.data.Dataset` representing `numpy_input`.
    """
        return self.extended.experimental_make_numpy_dataset(numpy_input, session=session)

    @deprecated(None, 'This method is not available in TF 2.x. Please switch to using `run` instead.')
    def experimental_run(self, fn, input_iterator=None):
        """Runs ops in `fn` on each replica, with inputs from `input_iterator`.

    DEPRECATED: This method is not available in TF 2.x. Please switch
    to using `run` instead.

    When eager execution is enabled, executes ops specified by `fn` on each
    replica. Otherwise, builds a graph to execute the ops on each replica.

    Each replica will take a single, different input from the inputs provided by
    one `get_next` call on the input iterator.

    `fn` may call `tf.distribute.get_replica_context()` to access members such
    as `replica_id_in_sync_group`.

    IMPORTANT: Depending on the `tf.distribute.Strategy` implementation being
    used, and whether eager execution is enabled, `fn` may be called one or more
    times (once for each replica).

    Args:
      fn: The function to run. The inputs to the function must match the outputs
        of `input_iterator.get_next()`. The output must be a `tf.nest` of
        `Tensor`s.
      input_iterator: (Optional) input iterator from which the inputs are taken.

    Returns:
      Merged return value of `fn` across replicas. The structure of the return
      value is the same as the return value from `fn`. Each element in the
      structure can either be `PerReplica` (if the values are unsynchronized),
      `Mirrored` (if the values are kept in sync), or `Tensor` (if running on a
      single replica).
    """
        return super(StrategyV1, self).experimental_run(fn, input_iterator)

    def reduce(self, reduce_op, value, axis=None):
        return super(StrategyV1, self).reduce(reduce_op, value, axis)
    reduce.__doc__ = StrategyBase.reduce.__doc__

    def update_config_proto(self, config_proto):
        """Returns a copy of `config_proto` modified for use with this strategy.

    DEPRECATED: This method is not available in TF 2.x.

    The updated config has something needed to run a strategy, e.g.
    configuration to run collective ops, or device filters to improve
    distributed training performance.

    Args:
      config_proto: a `tf.ConfigProto` object.

    Returns:
      The updated copy of the `config_proto`.
    """
        return self._extended._update_config_proto(config_proto)