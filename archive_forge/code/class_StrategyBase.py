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
class StrategyBase(object):
    """A state & compute distribution policy on a list of devices.

  See [the guide](https://www.tensorflow.org/guide/distributed_training)
  for overview and examples. See `tf.distribute.StrategyExtended` and
  [`tf.distribute`](https://www.tensorflow.org/api_docs/python/tf/distribute)
  for a glossary of concepts mentioned on this page such as "per-replica",
  _replica_, and _reduce_.

  In short:

  * To use it with Keras `compile`/`fit`,
    [please
    read](https://www.tensorflow.org/guide/distributed_training#using_tfdistributestrategy_with_keras).
  * You may pass descendant of `tf.distribute.Strategy` to
    `tf.estimator.RunConfig` to specify how a `tf.estimator.Estimator`
    should distribute its computation. See
    [guide](https://www.tensorflow.org/guide/distributed_training#using_tfdistributestrategy_with_estimator_limited_support).
  * Otherwise, use `tf.distribute.Strategy.scope` to specify that a
    strategy should be used when building an executing your model.
    (This puts you in the "cross-replica context" for this strategy, which
    means the strategy is put in control of things like variable placement.)
  * If you are writing a custom training loop, you will need to call a few more
    methods,
    [see the
    guide](https://www.tensorflow.org/guide/distributed_training#using_tfdistributestrategy_with_custom_training_loops):

      * Start by creating a `tf.data.Dataset` normally.
      * Use `tf.distribute.Strategy.experimental_distribute_dataset` to convert
        a `tf.data.Dataset` to something that produces "per-replica" values.
        If you want to manually specify how the dataset should be partitioned
        across replicas, use
        `tf.distribute.Strategy.distribute_datasets_from_function`
        instead.
      * Use `tf.distribute.Strategy.run` to run a function
        once per replica, taking values that may be "per-replica" (e.g.
        from a `tf.distribute.DistributedDataset` object) and returning
        "per-replica" values.
        This function is executed in "replica context", which means each
        operation is performed separately on each replica.
      * Finally use a method (such as `tf.distribute.Strategy.reduce`) to
        convert the resulting "per-replica" values into ordinary `Tensor`s.

  A custom training loop can be as simple as:

  ```
  with my_strategy.scope():
    @tf.function
    def distribute_train_epoch(dataset):
      def replica_fn(input):
        # process input and return result
        return result

      total_result = 0
      for x in dataset:
        per_replica_result = my_strategy.run(replica_fn, args=(x,))
        total_result += my_strategy.reduce(tf.distribute.ReduceOp.SUM,
                                           per_replica_result, axis=None)
      return total_result

    dist_dataset = my_strategy.experimental_distribute_dataset(dataset)
    for _ in range(EPOCHS):
      train_result = distribute_train_epoch(dist_dataset)
  ```

  This takes an ordinary `dataset` and `replica_fn` and runs it
  distributed using a particular `tf.distribute.Strategy` named
  `my_strategy` above. Any variables created in `replica_fn` are created
  using `my_strategy`'s policy, and library functions called by
  `replica_fn` can use the `get_replica_context()` API to implement
  distributed-specific behavior.

  You can use the `reduce` API to aggregate results across replicas and use
  this as a return value from one iteration over a
  `tf.distribute.DistributedDataset`. Or
  you can use `tf.keras.metrics` (such as loss, accuracy, etc.) to
  accumulate metrics across steps in a given epoch.

  See the
  [custom training loop
  tutorial](https://www.tensorflow.org/tutorials/distribute/custom_training)
  for a more detailed example.

  Note: `tf.distribute.Strategy` currently does not support TensorFlow's
  partitioned variables (where a single variable is split across multiple
  devices) at this time.
  """

    def __init__(self, extended):
        self._extended = extended
        self._scale_loss_for_estimator = False
        if not hasattr(extended, '_retrace_functions_for_each_device'):
            try:
                extended._retrace_functions_for_each_device = len(extended.worker_devices) > 1
                distribution_strategy_replica_gauge.get_cell('num_replicas').set(self.num_replicas_in_sync)
            except:
                extended._retrace_functions_for_each_device = True
        self._mean_reduce_helper_fns = {}
        self._reduce_sum_fns = {}
        self._should_use_with_coordinator = False

    @property
    def extended(self):
        """`tf.distribute.StrategyExtended` with additional methods."""
        return self._extended

    @tf_contextlib.contextmanager
    def _scale_loss_for_estimator_enabled(self):
        """Scope which sets a flag used for scaling losses in optimizer.

    Yields:
      `_scale_loss_for_estimator_enabled` is a context manager with a
      side effect, but doesn't return a value.
    """
        self._scale_loss_for_estimator = True
        try:
            yield
        finally:
            self._scale_loss_for_estimator = False

    def scope(self):
        """Context manager to make the strategy current and distribute variables.

    This method returns a context manager, and is used as follows:

    >>> strategy = tf.distribute.MirroredStrategy(["GPU:0", "GPU:1"])
    >>> # Variable created inside scope:
    >>> with strategy.scope():
    ...   mirrored_variable = tf.Variable(1.)
    >>> mirrored_variable
    MirroredVariable:{
      0: <tf.Variable 'Variable:0' shape=() dtype=float32, numpy=1.0>,
      1: <tf.Variable 'Variable/replica_1:0' shape=() dtype=float32, numpy=1.0>
    }
    >>> # Variable created outside scope:
    >>> regular_variable = tf.Variable(1.)
    >>> regular_variable
    <tf.Variable 'Variable:0' shape=() dtype=float32, numpy=1.0>

    _What happens when Strategy.scope is entered?_

    * `strategy` is installed in the global context as the "current" strategy.
      Inside this scope, `tf.distribute.get_strategy()` will now return this
      strategy. Outside this scope, it returns the default no-op strategy.
    * Entering the scope also enters the "cross-replica context". See
      `tf.distribute.StrategyExtended` for an explanation on cross-replica and
      replica contexts.
    * Variable creation inside `scope` is intercepted by the strategy. Each
      strategy defines how it wants to affect the variable creation. Sync
      strategies like `MirroredStrategy`, `TPUStrategy` and
      `MultiWorkerMiroredStrategy` create variables replicated on each replica,
      whereas `ParameterServerStrategy` creates variables on the parameter
      servers. This is done using a custom `tf.variable_creator_scope`.
    * In some strategies, a default device scope may also be entered: in
      `MultiWorkerMiroredStrategy`, a default device scope of "/CPU:0" is
      entered on each worker.

    Note: Entering a scope does not automatically distribute a computation, except
      in the case of high level training framework like keras `model.fit`. If
      you're not using `model.fit`, you
      need to use `strategy.run` API to explicitly distribute that computation.
      See an example in the [custom training loop tutorial](https://www.tensorflow.org/tutorials/distribute/custom_training).


    _What should be in scope and what should be outside?_

    There are a number of requirements on what needs to happen inside the scope.
    However, in places where we have information about which strategy is in use,
    we often enter the scope for the user, so they don't have to do it
    explicitly (i.e. calling those either inside or outside the scope is OK).

    * Anything that creates variables that should be distributed variables
      must be called in a `strategy.scope`. This can be accomplished either by
      directly calling the variable creating function within the scope context,
      or by relying on another API like `strategy.run` or `keras.Model.fit` to
      automatically enter it for you. Any variable that is created outside scope
      will not be distributed and may have performance implications. Some common
      objects that create variables in TF are Models, Optimizers, Metrics. Such
      objects should always be initialized in the scope, and any functions
      that may lazily create variables (e.g., `Model.__call__()`, tracing a
      `tf.function`, etc.) should similarly be called within scope. Another
      source of variable creation can be a checkpoint restore - when variables
      are created lazily. Note that any variable created inside a strategy
      captures the strategy information. So reading and writing to these
      variables outside the `strategy.scope` can also work seamlessly, without
      the user having to enter the scope.
    * Some strategy APIs (such as `strategy.run` and `strategy.reduce`) which
      require to be in a strategy's scope, enter the scope automatically, which
      means when using those APIs you don't need to explicitly enter the scope
      yourself.
    * When a `tf.keras.Model` is created inside a `strategy.scope`, the Model
      object captures the scope information. When high level training framework
      methods such as `model.compile`, `model.fit`, etc. are then called, the
      captured scope will be automatically entered, and the associated strategy
      will be used to distribute the training etc. See a detailed example in
      [distributed keras tutorial](https://www.tensorflow.org/tutorials/distribute/keras).
      WARNING: Simply calling `model(..)` does not automatically enter the
      captured scope -- only high level training framework APIs support this
      behavior: `model.compile`, `model.fit`, `model.evaluate`, `model.predict`
      and `model.save` can all be called inside or outside the scope.
    * The following can be either inside or outside the scope:
        * Creating the input datasets
        * Defining `tf.function`s that represent your training step
        * Saving APIs such as `tf.saved_model.save`. Loading creates variables,
          so that should go inside the scope if you want to train the model in a
          distributed way.
        * Checkpoint saving. As mentioned above - `checkpoint.restore` may
          sometimes need to be inside scope if it creates variables.

    Returns:
      A context manager.
    """
        return self._extended._scope(self)

    @doc_controls.do_not_doc_inheritable
    @deprecated(None, 'use extended.colocate_vars_with() instead.')
    def colocate_vars_with(self, colocate_with_variable):
        """DEPRECATED: use extended.colocate_vars_with() instead."""
        return self._extended.colocate_vars_with(colocate_with_variable)

    @doc_controls.do_not_generate_docs
    def make_dataset_iterator(self, dataset):
        """DEPRECATED TF 1.x ONLY."""
        return self._extended._make_dataset_iterator(dataset)

    @doc_controls.do_not_generate_docs
    def make_input_fn_iterator(self, input_fn, replication_mode=InputReplicationMode.PER_WORKER):
        """DEPRECATED TF 1.x ONLY."""
        if replication_mode != InputReplicationMode.PER_WORKER:
            raise ValueError('Input replication mode not supported: %r' % replication_mode)
        with self.scope():
            return self.extended._make_input_fn_iterator(input_fn, replication_mode=replication_mode)

    @doc_controls.do_not_generate_docs
    @deprecated(None, 'use run() instead')
    def experimental_run(self, fn, input_iterator=None):
        """DEPRECATED TF 1.x ONLY."""
        with self.scope():
            args = (input_iterator.get_next(),) if input_iterator is not None else ()
        return self.run(fn, args=args)

    def experimental_distribute_dataset(self, dataset, options=None):
        """Creates `tf.distribute.DistributedDataset` from `tf.data.Dataset`.

    The returned `tf.distribute.DistributedDataset` can be iterated over
    similar to regular datasets.
    NOTE: The user cannot add any more transformations to a
    `tf.distribute.DistributedDataset`. You can only create an iterator or
    examine the `tf.TypeSpec` of the data generated by it. See API docs of
    `tf.distribute.DistributedDataset` to learn more.

    The following is an example:

    >>> global_batch_size = 2
    >>> # Passing the devices is optional.
    ... strategy = tf.distribute.MirroredStrategy(devices=["GPU:0", "GPU:1"])
    >>> # Create a dataset
    ... dataset = tf.data.Dataset.range(4).batch(global_batch_size)
    >>> # Distribute that dataset
    ... dist_dataset = strategy.experimental_distribute_dataset(dataset)
    >>> @tf.function
    ... def replica_fn(input):
    ...   return input*2
    >>> result = []
    >>> # Iterate over the `tf.distribute.DistributedDataset`
    ... for x in dist_dataset:
    ...   # process dataset elements
    ...   result.append(strategy.run(replica_fn, args=(x,)))
    >>> print(result)
    [PerReplica:{
      0: <tf.Tensor: shape=(1,), dtype=int64, numpy=array([0])>,
      1: <tf.Tensor: shape=(1,), dtype=int64, numpy=array([2])>
    }, PerReplica:{
      0: <tf.Tensor: shape=(1,), dtype=int64, numpy=array([4])>,
      1: <tf.Tensor: shape=(1,), dtype=int64, numpy=array([6])>
    }]


    Three key actions happening under the hood of this method are batching,
    sharding, and prefetching.

    In the code snippet above, `dataset` is batched by `global_batch_size`, and
    calling `experimental_distribute_dataset` on it rebatches `dataset` to a
    new batch size that is equal to the global batch size divided by the number
    of replicas in sync. We iterate through it using a Pythonic for loop.
    `x` is a `tf.distribute.DistributedValues` containing data for all replicas,
    and each replica gets data of the new batch size.
    `tf.distribute.Strategy.run` will take care of feeding the right per-replica
    data in `x` to the right `replica_fn` executed on each replica.

    Sharding contains autosharding across multiple workers and within every
    worker. First, in multi-worker distributed training (i.e. when you use
    `tf.distribute.experimental.MultiWorkerMirroredStrategy`
    or `tf.distribute.TPUStrategy`), autosharding a dataset over a set of
    workers means that each worker is assigned a subset of the entire dataset
    (if the right `tf.data.experimental.AutoShardPolicy` is set). This is to
    ensure that at each step, a global batch size of non-overlapping dataset
    elements will be processed by each worker. Autosharding has a couple of
    different options that can be specified using
    `tf.data.experimental.DistributeOptions`. Then, sharding within each worker
    means the method will split the data among all the worker devices (if more
    than one a present). This will happen regardless of multi-worker
    autosharding.

    Note: for autosharding across multiple workers, the default mode is
    `tf.data.experimental.AutoShardPolicy.AUTO`. This mode
    will attempt to shard the input dataset by files if the dataset is
    being created out of reader datasets (e.g. `tf.data.TFRecordDataset`,
    `tf.data.TextLineDataset`, etc.) or otherwise shard the dataset by data,
    where each of the workers will read the entire dataset and only process the
    shard assigned to it. However, if you have less than one input file per
    worker, we suggest that you disable dataset autosharding across workers by
    setting the `tf.data.experimental.DistributeOptions.auto_shard_policy` to be
    `tf.data.experimental.AutoShardPolicy.OFF`.

    By default, this method adds a prefetch transformation at the end of the
    user provided `tf.data.Dataset` instance. The argument to the prefetch
    transformation which is `buffer_size` is equal to the number of replicas in
    sync.

    If the above batch splitting and dataset sharding logic is undesirable,
    please use
    `tf.distribute.Strategy.distribute_datasets_from_function`
    instead, which does not do any automatic batching or sharding for you.

    Note: If you are using TPUStrategy, the order in which the data is processed
    by the workers when using
    `tf.distribute.Strategy.experimental_distribute_dataset` or
    `tf.distribute.Strategy.distribute_datasets_from_function` is
    not guaranteed. This is typically required if you are using
    `tf.distribute` to scale prediction. You can however insert an index for
    each element in the batch and order outputs accordingly. Refer to [this
    snippet](https://www.tensorflow.org/tutorials/distribute/input#caveats)
    for an example of how to order outputs.

    Note: Stateful dataset transformations are currently not supported with
    `tf.distribute.experimental_distribute_dataset` or
    `tf.distribute.distribute_datasets_from_function`. Any stateful
    ops that the dataset may have are currently ignored. For example, if your
    dataset has a `map_fn` that uses `tf.random.uniform` to rotate an image,
    then you have a dataset graph that depends on state (i.e the random seed) on
    the local machine where the python process is being executed.

    For a tutorial on more usage and properties of this method, refer to the
    [tutorial on distributed input](https://www.tensorflow.org/tutorials/distribute/input#tfdistributestrategyexperimental_distribute_dataset).
    If you are interested in last partial batch handling, read [this section](https://www.tensorflow.org/tutorials/distribute/input#partial_batches).

    Args:
      dataset: `tf.data.Dataset` that will be sharded across all replicas using
        the rules stated above.
      options: `tf.distribute.InputOptions` used to control options on how this
        dataset is distributed.

    Returns:
      A `tf.distribute.DistributedDataset`.
    """
        distribution_strategy_input_api_counter.get_cell(self.__class__.__name__, 'distribute_dataset').increase_by(1)
        return self._extended._experimental_distribute_dataset(dataset, options)

    def distribute_datasets_from_function(self, dataset_fn, options=None):
        """Distributes `tf.data.Dataset` instances created by calls to `dataset_fn`.

    The argument `dataset_fn` that users pass in is an input function that has a
    `tf.distribute.InputContext` argument and returns a `tf.data.Dataset`
    instance. It is expected that the returned dataset from `dataset_fn` is
    already batched by per-replica batch size (i.e. global batch size divided by
    the number of replicas in sync) and sharded.
    `tf.distribute.Strategy.distribute_datasets_from_function` does
    not batch or shard the `tf.data.Dataset` instance
    returned from the input function. `dataset_fn` will be called on the CPU
    device of each of the workers and each generates a dataset where every
    replica on that worker will dequeue one batch of inputs (i.e. if a worker
    has two replicas, two batches will be dequeued from the `Dataset` every
    step).

    This method can be used for several purposes. First, it allows you to
    specify your own batching and sharding logic. (In contrast,
    `tf.distribute.experimental_distribute_dataset` does batching and sharding
    for you.) For example, where
    `experimental_distribute_dataset` is unable to shard the input files, this
    method might be used to manually shard the dataset (avoiding the slow
    fallback behavior in `experimental_distribute_dataset`). In cases where the
    dataset is infinite, this sharding can be done by creating dataset replicas
    that differ only in their random seed.

    The `dataset_fn` should take an `tf.distribute.InputContext` instance where
    information about batching and input replication can be accessed.

    You can use `element_spec` property of the
    `tf.distribute.DistributedDataset` returned by this API to query the
    `tf.TypeSpec` of the elements returned by the iterator. This can be used to
    set the `input_signature` property of a `tf.function`. Follow
    `tf.distribute.DistributedDataset.element_spec` to see an example.

    IMPORTANT: The `tf.data.Dataset` returned by `dataset_fn` should have a
    per-replica batch size, unlike `experimental_distribute_dataset`, which uses
    the global batch size. This may be computed using
    `input_context.get_per_replica_batch_size`.

    Note: If you are using TPUStrategy, the order in which the data is processed
    by the workers when using
    `tf.distribute.Strategy.experimental_distribute_dataset` or
    `tf.distribute.Strategy.distribute_datasets_from_function` is
    not guaranteed. This is typically required if you are using
    `tf.distribute` to scale prediction. You can however insert an index for
    each element in the batch and order outputs accordingly. Refer to [this
    snippet](https://www.tensorflow.org/tutorials/distribute/input#caveats)
    for an example of how to order outputs.

    Note: Stateful dataset transformations are currently not supported with
    `tf.distribute.experimental_distribute_dataset` or
    `tf.distribute.distribute_datasets_from_function`. Any stateful
    ops that the dataset may have are currently ignored. For example, if your
    dataset has a `map_fn` that uses `tf.random.uniform` to rotate an image,
    then you have a dataset graph that depends on state (i.e the random seed) on
    the local machine where the python process is being executed.

    For a tutorial on more usage and properties of this method, refer to the
    [tutorial on distributed input](https://www.tensorflow.org/tutorials/distribute/input#tfdistributestrategyexperimental_distribute_datasets_from_function)).
    If you are interested in last partial batch handling, read [this section](https://www.tensorflow.org/tutorials/distribute/input#partial_batches).

    Args:
      dataset_fn: A function taking a `tf.distribute.InputContext` instance and
        returning a `tf.data.Dataset`.
      options: `tf.distribute.InputOptions` used to control options on how this
        dataset is distributed.

    Returns:
      A `tf.distribute.DistributedDataset`.
    """
        distribution_strategy_input_api_counter.get_cell(self.__class__.__name__, 'distribute_datasets_from_function').increase_by(1)
        return self._extended._distribute_datasets_from_function(dataset_fn, options)

    @doc_controls.do_not_doc_inheritable
    @deprecation.deprecated(None, 'rename to distribute_datasets_from_function')
    def experimental_distribute_datasets_from_function(self, dataset_fn, options=None):
        return self.distribute_datasets_from_function(dataset_fn, options)

    def run(self, fn, args=(), kwargs=None, options=None):
        """Invokes `fn` on each replica, with the given arguments.

    This method is the primary way to distribute your computation with a
    tf.distribute object. It invokes `fn` on each replica. If `args` or `kwargs`
    have `tf.distribute.DistributedValues`, such as those produced by a
    `tf.distribute.DistributedDataset` from
    `tf.distribute.Strategy.experimental_distribute_dataset` or
    `tf.distribute.Strategy.distribute_datasets_from_function`,
    when `fn` is executed on a particular replica, it will be executed with the
    component of `tf.distribute.DistributedValues` that correspond to that
    replica.

    `fn` is invoked under a replica context. `fn` may call
    `tf.distribute.get_replica_context()` to access members such as
    `all_reduce`. Please see the module-level docstring of tf.distribute for the
    concept of replica context.

    All arguments in `args` or `kwargs` can be a nested structure of tensors,
    e.g. a list of tensors, in which case `args` and `kwargs` will be passed to
    the `fn` invoked on each replica. Or `args` or `kwargs` can be
    `tf.distribute.DistributedValues` containing tensors or composite tensors,
    i.e. `tf.compat.v1.TensorInfo.CompositeTensor`, in which case each `fn` call
    will get the component of a `tf.distribute.DistributedValues` corresponding
    to its replica. Note that arbitrary Python values that are not of the types
    above are not supported.

    IMPORTANT: Depending on the implementation of `tf.distribute.Strategy` and
    whether eager execution is enabled, `fn` may be called one or more times. If
    `fn` is annotated with `tf.function` or `tf.distribute.Strategy.run` is
    called inside a `tf.function` (eager execution is disabled inside a
    `tf.function` by default), `fn` is called once per replica to generate a
    Tensorflow graph, which will then be reused for execution with new inputs.
    Otherwise, if eager execution is enabled, `fn` will be called once per
    replica every step just like regular python code.

    Example usage:

    1.  Constant tensor input.

        >>> strategy = tf.distribute.MirroredStrategy(["GPU:0", "GPU:1"])
        >>> tensor_input = tf.constant(3.0)
        >>> @tf.function
        ... def replica_fn(input):
        ...   return input*2.0
        >>> result = strategy.run(replica_fn, args=(tensor_input,))
        >>> result
        PerReplica:{
          0: <tf.Tensor: shape=(), dtype=float32, numpy=6.0>,
          1: <tf.Tensor: shape=(), dtype=float32, numpy=6.0>
        }

    2.  DistributedValues input.  {: value=2}

        >>> strategy = tf.distribute.MirroredStrategy(["GPU:0", "GPU:1"])
        >>> @tf.function
        ... def run():
        ...   def value_fn(value_context):
        ...     return value_context.num_replicas_in_sync
        ...   distributed_values = (
        ...     strategy.experimental_distribute_values_from_function(
        ...       value_fn))
        ...   def replica_fn2(input):
        ...     return input*2
        ...   return strategy.run(replica_fn2, args=(distributed_values,))
        >>> result = run()
        >>> result
        <tf.Tensor: shape=(), dtype=int32, numpy=4>

    3.  Use `tf.distribute.ReplicaContext` to allreduce values. {: value=3}

        >>> strategy = tf.distribute.MirroredStrategy(["gpu:0", "gpu:1"])
        >>> @tf.function
        ... def run():
        ...    def value_fn(value_context):
        ...      return tf.constant(value_context.replica_id_in_sync_group)
        ...    distributed_values = (
        ...        strategy.experimental_distribute_values_from_function(
        ...            value_fn))
        ...    def replica_fn(input):
        ...      return tf.distribute.get_replica_context().all_reduce(
        ...          "sum", input)
        ...    return strategy.run(replica_fn, args=(distributed_values,))
        >>> result = run()
        >>> result
        PerReplica:{
          0: <tf.Tensor: shape=(), dtype=int32, numpy=1>,
          1: <tf.Tensor: shape=(), dtype=int32, numpy=1>
        }

    Args:
      fn: The function to run on each replica.
      args: Optional positional arguments to `fn`. Its element can be a tensor,
        a nested structure of tensors or a `tf.distribute.DistributedValues`.
      kwargs: Optional keyword arguments to `fn`. Its element can be a tensor,
        a nested structure of tensors or a `tf.distribute.DistributedValues`.
      options: An optional instance of `tf.distribute.RunOptions` specifying
        the options to run `fn`.

    Returns:
      Merged return value of `fn` across replicas. The structure of the return
      value is the same as the return value from `fn`. Each element in the
      structure can either be `tf.distribute.DistributedValues`, `Tensor`
      objects, or `Tensor`s (for example, if running on a single replica).
    """
        del options
        if not isinstance(args, (list, tuple)):
            raise ValueError('positional args must be a list or tuple, got {}'.format(type(args)))
        with self.scope():
            fn = autograph.tf_convert(fn, autograph_ctx.control_status_ctx(), convert_by_default=False)
            return self._extended.call_for_each_replica(fn, args=args, kwargs=kwargs)

    def reduce(self, reduce_op, value, axis):
        """Reduce `value` across replicas and return result on current device.

    >>> strategy = tf.distribute.MirroredStrategy(["GPU:0", "GPU:1"])
    >>> def step_fn():
    ...   i = tf.distribute.get_replica_context().replica_id_in_sync_group
    ...   return tf.identity(i)
    >>>
    >>> per_replica_result = strategy.run(step_fn)
    >>> total = strategy.reduce("SUM", per_replica_result, axis=None)
    >>> total
    <tf.Tensor: shape=(), dtype=int32, numpy=1>

    To see how this would look with multiple replicas, consider the same
    example with MirroredStrategy with 2 GPUs:

    ```python
    strategy = tf.distribute.MirroredStrategy(devices=["GPU:0", "GPU:1"])
    def step_fn():
      i = tf.distribute.get_replica_context().replica_id_in_sync_group
      return tf.identity(i)

    per_replica_result = strategy.run(step_fn)
    # Check devices on which per replica result is:
    strategy.experimental_local_results(per_replica_result)[0].device
    # /job:localhost/replica:0/task:0/device:GPU:0
    strategy.experimental_local_results(per_replica_result)[1].device
    # /job:localhost/replica:0/task:0/device:GPU:1

    total = strategy.reduce("SUM", per_replica_result, axis=None)
    # Check device on which reduced result is:
    total.device
    # /job:localhost/replica:0/task:0/device:CPU:0

    ```

    This API is typically used for aggregating the results returned from
    different replicas, for reporting etc. For example, loss computed from
    different replicas can be averaged using this API before printing.

    Note: The result is copied to the "current" device - which would typically
    be the CPU of the worker on which the program is running. For `TPUStrategy`,
    it is the first TPU host. For multi client `MultiWorkerMirroredStrategy`,
    this is CPU of each worker.

    There are a number of different tf.distribute APIs for reducing values
    across replicas:
    * `tf.distribute.ReplicaContext.all_reduce`: This differs from
    `Strategy.reduce` in that it is for replica context and does
    not copy the results to the host device. `all_reduce` should be typically
    used for reductions inside the training step such as gradients.
    * `tf.distribute.StrategyExtended.reduce_to` and
    `tf.distribute.StrategyExtended.batch_reduce_to`: These APIs are more
    advanced versions of `Strategy.reduce` as they allow customizing the
    destination of the result. They are also called in cross replica context.

    _What should axis be?_

    Given a per-replica value returned by `run`, say a
    per-example loss, the batch will be divided across all the replicas.  This
    function allows you to aggregate across replicas and optionally also across
    batch elements by specifying the axis parameter accordingly.

    For example, if you have a global batch size of 8 and 2
    replicas, values for examples `[0, 1, 2, 3]` will be on replica 0 and
    `[4, 5, 6, 7]` will be on replica 1. With `axis=None`, `reduce` will
    aggregate only across replicas, returning `[0+4, 1+5, 2+6, 3+7]`.
    This is useful when each replica is computing a scalar or some other value
    that doesn't have a "batch" dimension (like a gradient or loss).
    ```
    strategy.reduce("sum", per_replica_result, axis=None)
    ```

    Sometimes, you will want to aggregate across both the global batch _and_
    all replicas. You can get this behavior by specifying the batch
    dimension as the `axis`, typically `axis=0`. In this case it would return a
    scalar `0+1+2+3+4+5+6+7`.
    ```
    strategy.reduce("sum", per_replica_result, axis=0)
    ```

    If there is a last partial batch, you will need to specify an axis so
    that the resulting shape is consistent across replicas. So if the last
    batch has size 6 and it is divided into [0, 1, 2, 3] and [4, 5], you
    would get a shape mismatch unless you specify `axis=0`. If you specify
    `tf.distribute.ReduceOp.MEAN`, using `axis=0` will use the correct
    denominator of 6. Contrast this with computing `reduce_mean` to get a
    scalar value on each replica and this function to average those means,
    which will weigh some values `1/8` and others `1/4`.

    Args:
      reduce_op: a `tf.distribute.ReduceOp` value specifying how values should
        be combined. Allows using string representation of the enum such as
        "SUM", "MEAN".
      value: a `tf.distribute.DistributedValues` instance, e.g. returned by
        `Strategy.run`, to be combined into a single tensor. It can also be a
        regular tensor when used with `OneDeviceStrategy` or default strategy.
      axis: specifies the dimension to reduce along within each
        replica's tensor. Should typically be set to the batch dimension, or
        `None` to only reduce across replicas (e.g. if the tensor has no batch
        dimension).

    Returns:
      A `Tensor`.
    """
        _require_cross_replica_or_default_context_extended(self._extended)
        if isinstance(reduce_op, six.string_types):
            reduce_op = reduce_util.ReduceOp(reduce_op.upper())
        if axis is None:
            return self._extended._reduce(reduce_op, value)
        if reduce_op == reduce_util.ReduceOp.SUM:

            def reduce_sum(v):
                return math_ops.reduce_sum(v, axis=axis)
            if eager_context.executing_eagerly():
                if axis not in self._reduce_sum_fns:
                    self._reduce_sum_fns[axis] = def_function.function(reduce_sum)
                value = self.run(self._reduce_sum_fns[axis], args=(value,))
            else:
                value = self.run(reduce_sum, args=(value,))
            return self._extended._reduce(reduce_op, value)
        if reduce_op != reduce_util.ReduceOp.MEAN:
            raise TypeError('Expected `reduce_op` to be a `tf.distribute.ReduceOp`, not: %r' % reduce_op)

        def mean_reduce_helper(v, axes=axis):
            """Computes the numerator and denominator on each replica."""
            numer = math_ops.reduce_sum(v, axis=axes)

            def dimension(axis):
                if v.shape.rank is not None:
                    if axis < 0:
                        if axis + v.shape.rank < 0:
                            raise ValueError('`axis` = %r out of range for `value` with rank %d' % (axis, v.shape.rank))
                        axis += v.shape.rank
                    elif axis >= v.shape.rank:
                        raise ValueError('`axis` = %r out of range for `value` with rank %d' % (axis, v.shape.rank))
                    dim = tensor_shape.dimension_value(v.shape[axis])
                    if dim is not None:
                        return array_ops.identity(constant_op.constant(dim, dtype=dtypes.int64))
                elif axis < 0:
                    axis = axis + array_ops.rank(v)
                return array_ops.identity(array_ops.shape_v2(v, out_type=dtypes.int64)[axis])
            if isinstance(axis, six.integer_types):
                denom = dimension(axis)
            elif isinstance(axis, (tuple, list)):
                denom = math_ops.reduce_prod([dimension(a) for a in axes])
            else:
                raise TypeError('Expected `axis` to be an integer, tuple or list not: %r' % axis)
            return (numer, denom)
        if eager_context.executing_eagerly():
            if axis not in self._mean_reduce_helper_fns:
                self._mean_reduce_helper_fns[axis] = def_function.function(mean_reduce_helper)
            numer, denom = self.run(self._mean_reduce_helper_fns[axis], args=(value,))
        else:
            numer, denom = self.run(mean_reduce_helper, args=(value,))
        numer = self._extended._reduce(reduce_util.ReduceOp.SUM, numer)
        denom = self._extended._reduce(reduce_util.ReduceOp.SUM, denom)
        denom = math_ops.cast(denom, numer.dtype)
        return math_ops.truediv(numer, denom)

    @doc_controls.do_not_doc_inheritable
    @deprecated(None, 'use `experimental_local_results` instead.')
    def unwrap(self, value):
        """Returns the list of all local per-replica values contained in `value`.

    DEPRECATED: Please use `experimental_local_results` instead.

    Note: This only returns values on the workers initiated by this client.
    When using a `tf.distribute.Strategy` like
    `tf.distribute.experimental.MultiWorkerMirroredStrategy`, each worker
    will be its own client, and this function will only return values
    computed on that worker.

    Args:
      value: A value returned by `experimental_run()`,
        `extended.call_for_each_replica()`, or a variable created in `scope`.

    Returns:
      A tuple of values contained in `value`. If `value` represents a single
      value, this returns `(value,).`
    """
        return self._extended._local_results(value)

    def experimental_local_results(self, value):
        """Returns the list of all local per-replica values contained in `value`.

    Note: This only returns values on the worker initiated by this client.
    When using a `tf.distribute.Strategy` like
    `tf.distribute.experimental.MultiWorkerMirroredStrategy`, each worker
    will be its own client, and this function will only return values
    computed on that worker.

    Args:
      value: A value returned by `experimental_run()`, `run(), or a variable
      created in `scope`.

    Returns:
      A tuple of values contained in `value` where ith element corresponds to
      ith replica. If `value` represents a single value, this returns
      `(value,).`
    """
        return self._extended._local_results(value)

    @doc_controls.do_not_doc_inheritable
    def group(self, value, name=None):
        """Shortcut for `tf.group(self.experimental_local_results(value))`."""
        return self._extended._group(value, name)

    @property
    def num_replicas_in_sync(self):
        """Returns number of replicas over which gradients are aggregated."""
        return self._extended._num_replicas_in_sync

    @doc_controls.do_not_doc_inheritable
    @deprecated(None, 'use `update_config_proto` instead.')
    def configure(self, session_config=None, cluster_spec=None, task_type=None, task_id=None):
        """DEPRECATED: use `update_config_proto` instead.

    Configures the strategy class.

    DEPRECATED: This method's functionality has been split into the strategy
    constructor and `update_config_proto`. In the future, we will allow passing
    cluster and config_proto to the constructor to configure the strategy. And
    `update_config_proto` can be used to update the config_proto based on the
    specific strategy.
    """
        return self._extended._configure(session_config, cluster_spec, task_type, task_id)

    @doc_controls.do_not_generate_docs
    def update_config_proto(self, config_proto):
        """DEPRECATED TF 1.x ONLY."""
        return self._extended._update_config_proto(config_proto)

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, copy.deepcopy(v, memo))
        result._extended._container_strategy_weakref = weakref.ref(result)
        return result

    def __copy__(self):
        raise RuntimeError('Must only deepcopy DistributionStrategy.')

    @property
    def cluster_resolver(self):
        """Returns the cluster resolver associated with this strategy.

    In general, when using a multi-worker `tf.distribute` strategy such as
    `tf.distribute.experimental.MultiWorkerMirroredStrategy` or
    `tf.distribute.TPUStrategy()`, there is a
    `tf.distribute.cluster_resolver.ClusterResolver` associated with the
    strategy used, and such an instance is returned by this property.

    Strategies that intend to have an associated
    `tf.distribute.cluster_resolver.ClusterResolver` must set the
    relevant attribute, or override this property; otherwise, `None` is returned
    by default. Those strategies should also provide information regarding what
    is returned by this property.

    Single-worker strategies usually do not have a
    `tf.distribute.cluster_resolver.ClusterResolver`, and in those cases this
    property will return `None`.

    The `tf.distribute.cluster_resolver.ClusterResolver` may be useful when the
    user needs to access information such as the cluster spec, task type or task
    id. For example,

    ```python

    os.environ['TF_CONFIG'] = json.dumps({
      'cluster': {
          'worker': ["localhost:12345", "localhost:23456"],
          'ps': ["localhost:34567"]
      },
      'task': {'type': 'worker', 'index': 0}
    })

    # This implicitly uses TF_CONFIG for the cluster and current task info.
    strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()

    ...

    if strategy.cluster_resolver.task_type == 'worker':
      # Perform something that's only applicable on workers. Since we set this
      # as a worker above, this block will run on this particular instance.
    elif strategy.cluster_resolver.task_type == 'ps':
      # Perform something that's only applicable on parameter servers. Since we
      # set this as a worker above, this block will not run on this particular
      # instance.
    ```

    For more information, please see
    `tf.distribute.cluster_resolver.ClusterResolver`'s API docstring.

    Returns:
      The cluster resolver associated with this strategy. Returns `None` if a
      cluster resolver is not applicable or available in this strategy.
    """
        if hasattr(self.extended, '_cluster_resolver'):
            return self.extended._cluster_resolver
        return None