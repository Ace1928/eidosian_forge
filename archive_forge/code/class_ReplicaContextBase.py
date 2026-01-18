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
class ReplicaContextBase(object):
    """A class with a collection of APIs that can be called in a replica context.

  You can use `tf.distribute.get_replica_context` to get an instance of
  `ReplicaContext`, which can only be called inside the function passed to
  `tf.distribute.Strategy.run`.

  >>> strategy = tf.distribute.MirroredStrategy(['GPU:0', 'GPU:1'])
  >>> def func():
  ...   replica_context = tf.distribute.get_replica_context()
  ...   return replica_context.replica_id_in_sync_group
  >>> strategy.run(func)
  PerReplica:{
    0: <tf.Tensor: shape=(), dtype=int32, numpy=0>,
    1: <tf.Tensor: shape=(), dtype=int32, numpy=1>
  }
  """

    def __init__(self, strategy, replica_id_in_sync_group):
        """Creates a ReplicaContext.

    Args:
      strategy: A `tf.distribute.Strategy`.
      replica_id_in_sync_group: An integer, a `Tensor` or None. Prefer an
        integer whenever possible to avoid issues with nested `tf.function`. It
        accepts a `Tensor` only to be compatible with `tpu.replicate`.
    """
        self._strategy = strategy
        self._thread_context = _InReplicaThreadMode(self)
        if not (replica_id_in_sync_group is None or tensor_util.is_tf_type(replica_id_in_sync_group) or isinstance(replica_id_in_sync_group, int)):
            raise ValueError('replica_id_in_sync_group can only be an integer, a Tensor or None.')
        self._replica_id_in_sync_group = replica_id_in_sync_group
        if strategy:
            self._local_replica_id = strategy.extended._get_local_replica_id(replica_id_in_sync_group)
        self._summary_recording_distribution_strategy = None

    @doc_controls.do_not_generate_docs
    def __enter__(self):
        _push_per_thread_mode(self._thread_context)

        def replica_id_is_zero():
            return math_ops.equal(self.replica_id_in_sync_group, constant_op.constant(0))
        summary_state = summary_ops_v2._summary_state
        self._summary_recording_distribution_strategy = summary_state.is_recording_distribution_strategy
        summary_state.is_recording_distribution_strategy = replica_id_is_zero

    @doc_controls.do_not_generate_docs
    def __exit__(self, exception_type, exception_value, traceback):
        summary_state = summary_ops_v2._summary_state
        summary_state.is_recording_distribution_strategy = self._summary_recording_distribution_strategy
        _pop_per_thread_mode()

    def merge_call(self, merge_fn, args=(), kwargs=None):
        """Merge args across replicas and run `merge_fn` in a cross-replica context.

    This allows communication and coordination when there are multiple calls
    to the step_fn triggered by a call to `strategy.run(step_fn, ...)`.

    See `tf.distribute.Strategy.run` for an explanation.

    If not inside a distributed scope, this is equivalent to:

    ```
    strategy = tf.distribute.get_strategy()
    with cross-replica-context(strategy):
      return merge_fn(strategy, *args, **kwargs)
    ```

    Args:
      merge_fn: Function that joins arguments from threads that are given as
        PerReplica. It accepts `tf.distribute.Strategy` object as
        the first argument.
      args: List or tuple with positional per-thread arguments for `merge_fn`.
      kwargs: Dict with keyword per-thread arguments for `merge_fn`.

    Returns:
      The return value of `merge_fn`, except for `PerReplica` values which are
      unpacked.
    """
        require_replica_context(self)
        if kwargs is None:
            kwargs = {}
        merge_fn = autograph.tf_convert(merge_fn, autograph_ctx.control_status_ctx(), convert_by_default=False)
        return self._merge_call(merge_fn, args, kwargs)

    def _merge_call(self, merge_fn, args, kwargs):
        """Default implementation for single replica."""
        _push_per_thread_mode(_CrossReplicaThreadMode(self._strategy))
        try:
            return merge_fn(self._strategy, *args, **kwargs)
        finally:
            _pop_per_thread_mode()

    @property
    def num_replicas_in_sync(self):
        """Returns number of replicas that are kept in sync."""
        return self._strategy.num_replicas_in_sync

    @property
    def replica_id_in_sync_group(self):
        """Returns the id of the replica.

    This identifies the replica among all replicas that are kept in sync. The
    value of the replica id can range from 0 to
    `tf.distribute.ReplicaContext.num_replicas_in_sync` - 1.

    NOTE: This is not guaranteed to be the same ID as the XLA replica ID use
    for low-level operations such as collective_permute.

    Returns:
      a `Tensor`.
    """
        if tensor_util.is_tf_type(self._replica_id_in_sync_group):
            return self._replica_id_in_sync_group
        return constant_op.constant(self._replica_id_in_sync_group, dtypes.int32, name='replica_id_in_sync_group')

    @property
    def _replica_id(self):
        """This is the local replica id in a given sync group."""
        return self._local_replica_id

    @property
    def strategy(self):
        """The current `tf.distribute.Strategy` object."""
        return self._strategy

    @property
    @deprecation.deprecated(None, 'Please avoid relying on devices property.')
    def devices(self):
        """Returns the devices this replica is to be executed on, as a tuple of strings.

    NOTE: For `tf.distribute.MirroredStrategy` and
    `tf.distribute.experimental.MultiWorkerMirroredStrategy`, this returns a
    nested
    list of device strings, e.g., [["GPU:0"]].
    """
        require_replica_context(self)
        return (device_util.current(),)

    def all_reduce(self, reduce_op, value, options=None):
        """All-reduces `value` across all replicas.

    >>> strategy = tf.distribute.MirroredStrategy(["GPU:0", "GPU:1"])
    >>> def step_fn():
    ...   ctx = tf.distribute.get_replica_context()
    ...   value = tf.identity(1.)
    ...   return ctx.all_reduce(tf.distribute.ReduceOp.SUM, value)
    >>> strategy.experimental_local_results(strategy.run(step_fn))
    (<tf.Tensor: shape=(), dtype=float32, numpy=2.0>,
     <tf.Tensor: shape=(), dtype=float32, numpy=2.0>)

    It supports batched operations. You can pass a list of values and it
    attempts to batch them when possible. You can also specify `options`
    to indicate the desired batching behavior, e.g. batch the values into
    multiple packs so that they can better overlap with computations.

    >>> strategy = tf.distribute.MirroredStrategy(["GPU:0", "GPU:1"])
    >>> def step_fn():
    ...   ctx = tf.distribute.get_replica_context()
    ...   value1 = tf.identity(1.)
    ...   value2 = tf.identity(2.)
    ...   return ctx.all_reduce(tf.distribute.ReduceOp.SUM, [value1, value2])
    >>> strategy.experimental_local_results(strategy.run(step_fn))
    ([<tf.Tensor: shape=(), dtype=float32, numpy=2.0>,
    <tf.Tensor: shape=(), dtype=float32, numpy=4.0>],
    [<tf.Tensor: shape=(), dtype=float32, numpy=2.0>,
    <tf.Tensor: shape=(), dtype=float32, numpy=4.0>])

    Note that all replicas need to participate in the all-reduce, otherwise this
    operation hangs. Note that if there're multiple all-reduces, they need to
    execute in the same order on all replicas. Dispatching all-reduce based on
    conditions is usually error-prone.

    Known limitation: if `value` contains `tf.IndexedSlices`, attempting to
    compute gradient w.r.t `value` would result in an error.

    This API currently can only be called in the replica context. Other
    variants to reduce values across replicas are:
    * `tf.distribute.StrategyExtended.reduce_to`: the reduce and all-reduce API
      in the cross-replica context.
    * `tf.distribute.StrategyExtended.batch_reduce_to`: the batched reduce and
      all-reduce API in the cross-replica context.
    * `tf.distribute.Strategy.reduce`: a more convenient method to reduce
      to the host in cross-replica context.

    Args:
      reduce_op: a `tf.distribute.ReduceOp` value specifying how values should
        be combined. Allows using string representation of the enum such as
        "SUM", "MEAN".
      value: a potentially nested structure of `tf.Tensor` or `tf.IndexedSlices` which
        `tf.nest.flatten` accepts. The structure and the shapes of `value` need to be
        same on all replicas.
      options: a `tf.distribute.experimental.CommunicationOptions`. Options to
        perform collective operations. This overrides the default options if the
        `tf.distribute.Strategy` takes one in the constructor. See
        `tf.distribute.experimental.CommunicationOptions` for details of the
        options.

    Returns:
       A nested structure of `tf.Tensor` with the reduced values. The structure
       is the same as `value`.
    """
        flattened_value = nest.flatten(value)
        has_indexed_slices = False
        for v in flattened_value:
            if isinstance(v, indexed_slices.IndexedSlices):
                has_indexed_slices = True
        if isinstance(reduce_op, six.string_types):
            reduce_op = reduce_util.ReduceOp(reduce_op.upper())
        if options is None:
            options = collective_util.Options()

        def batch_all_reduce(strategy, *value_flat):
            return strategy.extended.batch_reduce_to(reduce_op, [(v, _batch_reduce_destination(v)) for v in value_flat], options)
        if self._strategy.extended._use_merge_call():
            if has_indexed_slices:
                return nest.pack_sequence_as(value, self.merge_call(batch_all_reduce, args=flattened_value))

            @custom_gradient.custom_gradient
            def grad_wrapper(*xs):
                ys = self.merge_call(batch_all_reduce, args=xs)
                return (ys, lambda *dy_s: self.all_reduce(reduce_op, dy_s))
            return nest.pack_sequence_as(value, grad_wrapper(*flattened_value))
        else:
            if has_indexed_slices:
                return nest.pack_sequence_as(value, self._strategy.extended._replica_ctx_all_reduce(reduce_op, flattened_value, options))

            @custom_gradient.custom_gradient
            def grad_wrapper(*xs):
                ys = self._strategy.extended._replica_ctx_all_reduce(reduce_op, xs, options)
                return (ys, lambda *dy_s: self.all_reduce(reduce_op, dy_s))
            return nest.pack_sequence_as(value, grad_wrapper(*flattened_value))