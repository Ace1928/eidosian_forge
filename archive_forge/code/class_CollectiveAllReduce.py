import collections
import copy
import multiprocessing.dummy
import multiprocessing.pool
import threading
import numpy as np
import six
from tensorflow.python.client import device_lib
from tensorflow.python.distribute import collective_util
from tensorflow.python.distribute import cross_device_utils
from tensorflow.python.distribute import device_util
from tensorflow.python.distribute import distribute_utils
from tensorflow.python.distribute import ps_values
from tensorflow.python.distribute import reduce_util
from tensorflow.python.distribute import tpu_values
from tensorflow.python.distribute import values as value_lib
from tensorflow.python.distribute import values_util
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import kernels
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import tf_export
from tensorflow.tools.docs import doc_controls
class CollectiveAllReduce(CrossDeviceOps):
    """All-reduce cross device ops using collective ops.

  In the between-graph replicated training, it will still do all-reduces across
  all workers and then put results on the right destinations.
  """

    def __init__(self, devices, group_size, options, collective_keys=None, canonicalize_devices=True):
        """Initializes the object.

    Args:
      devices: a list of device strings to run collectives on.
      group_size: the global group size. For between-graph replicated training
        it's the total number of devices across all workers.
      options: a `tf.distribute.experimental.CommunicationOptions`.
      collective_keys: an optional CollectiveKey object.
      canonicalize_devices: Whether to canonicalize devices for workers or not.
    """
        if group_size % len(devices) > 0:
            raise ValueError('group_size must be divisible by the number of devices.')
        self._group_size = group_size
        self._options = options
        self._collective_keys = collective_keys or cross_device_utils.CollectiveKeys()
        self._lock = threading.Lock()
        if canonicalize_devices:
            self._devices = tuple((device_util.canonicalize(d) for d in devices))
        else:
            self._devices = tuple((device_util.canonicalize_without_job_and_task(d) for d in devices))
        group_key = self._collective_keys.get_group_key(self._devices)
        self._launchers = []
        self._limited_nccl = False
        for device in self._devices:
            launcher = cross_device_utils.CollectiveReplicaLauncher(group_key, group_size, self._collective_keys, device, options)
            self._launchers.append(launcher)
            if not launcher.can_order_nccl():
                self._limited_nccl = True
        super(CollectiveAllReduce, self).__init__()
        self._canonicalize_devices = canonicalize_devices

    @property
    def _num_between_graph_workers(self):
        return self._group_size / len(self._devices)

    def _all_reduce(self, reduce_op, value, replica_id, options):
        """Implements CrossDeviceOps.all_reduce."""
        flat_values = nest.flatten(value)
        if self._limited_nccl and options.implementation == collective_util.CommunicationImplementation.NCCL and (len(flat_values) == 1):
            options = options.merge(collective_util.Options(implementation=collective_util.CommunicationImplementation.RING))
        launcher = self._launchers[replica_id]
        dense_values, dense_indices, sparse_values, sparse_indices = cross_device_utils.split_by_sparsity(flat_values)
        dense_results = []
        sparse_results = []
        if dense_values:
            dense_values.reverse()
            packs = cross_device_utils.group_by_size(dense_values, options.bytes_per_pack)
            if not context.executing_eagerly() and replica_id == 0:
                logging.info('Collective all_reduce tensors: %d all_reduces, num_devices = %d, group_size = %d, implementation = %s, num_packs = %d', len(dense_values), len(self._launchers), self._group_size, options.implementation, len(packs))
            dense_results = launcher.batch_all_reduce(packs, options)
            if reduce_op == reduce_util.ReduceOp.MEAN:
                for i, v in enumerate(dense_results):
                    with ops.device(self._devices[replica_id]):
                        dense_results[i] = v / self._group_size
            dense_results.reverse()
        if sparse_values:
            if not context.executing_eagerly() and replica_id == 0:
                logging.info('Collective all_reduce IndexedSlices: %d all_reduces, num_devices =%d, group_size = %d, implementation = %s', len(sparse_values), len(self._launchers), self._group_size, options.implementation)
            for indexed_slice in sparse_values:
                sparse_results.append(launcher.all_reduce_indexed_slices(indexed_slice, options))
            if reduce_op == reduce_util.ReduceOp.MEAN:
                for i, v in enumerate(sparse_results):
                    with ops.device(self._devices[replica_id]):
                        sparse_results[i] = indexed_slices.IndexedSlices(values=sparse_results[i].values / self._group_size, indices=sparse_results[i].indices, dense_shape=sparse_results[i].dense_shape)
        flat_results = cross_device_utils.stitch_values(((dense_results, dense_indices), (sparse_results, sparse_indices)))
        return nest.pack_sequence_as(value, flat_results)

    def _all_reduce_per_replica_values(self, reduce_op, per_replica_values, options):
        """All reduce a list of per_replica_value."""
        values_by_device = [[] for _ in self._devices]
        num_devices = len(self._devices)
        for per_replica in per_replica_values:
            for i in range(num_devices):
                values_by_device[i].append(per_replica.values[i])
        if context.executing_eagerly():

            def thread_fn(device_id):
                with context.eager_mode():
                    return self._all_reduce(reduce_op, values_by_device[device_id], device_id, options)
            with self._lock:
                pool = multiprocessing.pool.ThreadPool(len(self._devices))
                outputs_by_device = pool.map(thread_fn, list(range(num_devices)))
                pool.close()
        else:
            outputs_by_device = []
            with self._lock:
                for i in range(num_devices):
                    outputs_by_device.append(self._all_reduce(reduce_op, values_by_device[i], i, options))
        result = []
        for values in zip(*outputs_by_device):
            result.append(distribute_utils.regroup(values, wrap_class=value_lib.Mirrored))
        return result

    def reduce_implementation(self, reduce_op, per_replica_value, destinations, options):
        values_util.mark_as_unsaveable()
        all_reduced = self._all_reduce_per_replica_values(reduce_op, [per_replica_value], options)[0]
        devices = get_devices_from(destinations, self._canonicalize_devices)
        if _devices_match(per_replica_value, destinations, self._canonicalize_devices):
            return all_reduced
        if not isinstance(all_reduced, value_lib.Mirrored):
            all_reduced = value_lib.Mirrored([all_reduced])
        index = []
        with ops.control_dependencies(all_reduced.values):
            for d in devices:
                with ops.device(d):
                    for v in all_reduced.values:
                        if v.device == d:
                            index.append(array_ops.identity(v))
                            break
                    else:
                        index.append(array_ops.identity(all_reduced._primary))
        return distribute_utils.regroup(index, wrap_class=value_lib.Mirrored)

    def batch_reduce_implementation(self, reduce_op, value_destination_pairs, options):
        values_util.mark_as_unsaveable()
        all_devices_match = _all_devices_match(value_destination_pairs, self._canonicalize_devices)
        if all_devices_match:
            return self._all_reduce_per_replica_values(reduce_op, [v[0] for v in value_destination_pairs], options)
        else:
            if not all_devices_match:
                logging.log_first_n(logging.WARN, 'Efficient batch_reduce is not supported if destinations are different.', 10)
            return [self.reduce_implementation(reduce_op, value, dest, options) for value, dest in value_destination_pairs]

    def _gather_implementation(self, per_replica_value, destinations, axis, options):
        all_gathered = self._batch_all_gather([per_replica_value], axis, options)[0]
        values_util.mark_as_unsaveable()
        devices = get_devices_from(destinations, self._canonicalize_devices)
        if _devices_match(per_replica_value, destinations, self._canonicalize_devices):
            return all_gathered
        if not isinstance(all_gathered, value_lib.Mirrored):
            all_gathered = value_lib.Mirrored([all_gathered])
        index = []
        with ops.control_dependencies(all_gathered.values):
            for d in devices:
                with ops.device(d):
                    for v in all_gathered.values:
                        if v.device == d:
                            index.append(array_ops.identity(v))
                            break
                        else:
                            index.append(array_ops.identity(all_gathered._primary))
        return distribute_utils.regroup(index, wrap_class=value_lib.Mirrored)

    def _batch_all_gather(self, per_replica_values, axis, options):
        """all gather multiple per-replica-values."""
        batch_size = len(per_replica_values)
        if self._limited_nccl and options.implementation == collective_util.CommunicationImplementation.NCCL and (batch_size == 1):
            options = options.merge(collective_util.Options(implementation=collective_util.CommunicationImplementation.RING))
        logging.log_first_n(logging.INFO, 'Collective batch_all_gather: %d all-gathers, num_devices = %d, group_size = %d, implementation = %s, ' % (batch_size, len(self._devices), self._group_size, options.implementation), 10)

        def compute_gathered_values():
            gathered_values = []
            with self._lock, ops.name_scope('allgather'):
                for per_replica in per_replica_values:
                    outputs = []
                    for i in range(len(self._devices)):
                        outputs.append(self._launchers[i].all_gather(per_replica.values[i], axis, options))
                    gathered_values.append(outputs)
            return gathered_values
        if context.executing_eagerly():
            gathered_values = def_function.function(compute_gathered_values)()
        else:
            gathered_values = compute_gathered_values()
        mirrored = []
        for value in gathered_values:
            mirrored.append(distribute_utils.regroup(value, wrap_class=value_lib.Mirrored))
        return mirrored

    def __deepcopy__(self, memo):
        collective_keys = copy.deepcopy(self._collective_keys, memo)
        return CollectiveAllReduce(self._devices, self._group_size, self._options, collective_keys, self._canonicalize_devices)