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
class TPUExtended(distribute_lib.StrategyExtendedV1):
    """Implementation of TPUStrategy."""

    def __init__(self, container_strategy, tpu_cluster_resolver=None, steps_per_run=None, device_assignment=None, use_spmd_for_xla_partitioning=False, enable_data_reorder=False):
        super(TPUExtended, self).__init__(container_strategy)
        if tpu_cluster_resolver is None:
            tpu_cluster_resolver = tpu_cluster_resolver_lib.TPUClusterResolver('')
        if steps_per_run is None:
            steps_per_run = 1
        self._tpu_function_cache = weakref.WeakKeyDictionary()
        self._tpu_cluster_resolver = tpu_cluster_resolver
        self._tpu_metadata = self._tpu_cluster_resolver.get_tpu_system_metadata()
        self._device_assignment = device_assignment
        tpu_devices_flat = [d.name for d in self._tpu_metadata.devices if 'device:TPU:' in d.name]
        if device_assignment is None:
            self._tpu_devices = np.array([[d] for d in tpu_devices_flat], dtype=object)
        else:
            job_name = device_spec.DeviceSpecV2.from_string(tpu_devices_flat[0]).job
            tpu_devices = []
            for replica_id in range(device_assignment.num_replicas):
                replica_devices = []
                for logical_core in range(device_assignment.num_cores_per_replica):
                    replica_devices.append(device_util.canonicalize(device_assignment.tpu_device(replica=replica_id, logical_core=logical_core, job=job_name)))
                tpu_devices.append(replica_devices)
            self._tpu_devices = np.array(tpu_devices, dtype=object)
        self._host_device = device_util.get_host_for_device(self._tpu_devices[0][0])
        self._device_input_worker_devices = collections.OrderedDict()
        self._host_input_worker_devices = collections.OrderedDict()
        for tpu_device in self._tpu_devices[:, 0]:
            host_device = device_util.get_host_for_device(tpu_device)
            self._device_input_worker_devices.setdefault(host_device, [])
            self._device_input_worker_devices[host_device].append(tpu_device)
            self._host_input_worker_devices.setdefault(host_device, [])
            self._host_input_worker_devices[host_device].append(host_device)
        self._replica_order = self._get_replica_order(self._tpu_devices[:, 0]) if enable_data_reorder else None
        self.steps_per_run = steps_per_run
        self._require_static_shapes = True
        self.experimental_enable_get_next_as_optional = True
        self._logical_device_stack = [0]
        if context.executing_eagerly():
            atexit.register(context.async_wait)
        self._use_var_policy = not use_spmd_for_xla_partitioning
        self._use_spmd_for_xla_partitioning = use_spmd_for_xla_partitioning
        self._using_custom_device = False
        devices = self._tpu_devices[:, self._logical_device_stack[-1]]
        for d in devices:
            if context.is_custom_device(d):
                self._using_custom_device = True
                break

    def _get_replica_order(self, tpu_devices):
        """Get the replica order based on the tpu device order.

    For example, if the tpu_devices are:
    '/job:worker/replica:0/task:0/device:TPU:0',
    '/job:worker/replica:0/task:0/device:TPU:2',
    '/job:worker/replica:0/task:1/device:TPU:0',
    '/job:worker/replica:0/task:1/device:TPU:2',
    '/job:worker/replica:0/task:1/device:TPU:6',
    '/job:worker/replica:0/task:1/device:TPU:4',
    '/job:worker/replica:0/task:0/device:TPU:6',
    '/job:worker/replica:0/task:0/device:TPU:4',

    the returned replica order will be:
    [0, 1, 7, 6, 2, 3, 5, 4]

    This replica order will be used to reorder the data returned by the
    iterators,
    so that they can be placed on the same node as their computation graphs.

    Args:
      tpu_devices (List[str]): A list of tpu device names in the order of
        replicas.

    Returns:
      A list containing the order ids of corresponding TPU devices.
    """
        devices_with_ids = []
        for i, tpu_device in enumerate(tpu_devices):
            spec = tf_device.DeviceSpec.from_string(tpu_device)
            devices_with_ids.append(((spec.job, spec.replica, spec.device_type, spec.task, spec.device_index), i))
        return [i for _, i in sorted(devices_with_ids)]

    def _validate_colocate_with_variable(self, colocate_with_variable):
        distribute_utils.validate_colocate(colocate_with_variable, self)

    def _make_dataset_iterator(self, dataset):
        """Make iterators for each of the TPU hosts."""
        input_workers = input_lib.InputWorkers(tuple(self._device_input_worker_devices.items()))
        return input_lib_v1.DatasetIterator(dataset, input_workers, self._container_strategy(), num_replicas_in_sync=self._num_replicas_in_sync)

    def _make_input_fn_iterator(self, input_fn, replication_mode=distribute_lib.InputReplicationMode.PER_WORKER):
        input_contexts = []
        input_workers = input_lib.InputWorkers(tuple(self._device_input_worker_devices.items()))
        num_workers = input_workers.num_workers
        for i in range(num_workers):
            input_contexts.append(distribute_lib.InputContext(num_input_pipelines=num_workers, input_pipeline_id=i, num_replicas_in_sync=self._num_replicas_in_sync))
        return input_lib_v1.InputFunctionIterator(input_fn, input_workers, input_contexts, self._container_strategy())

    def _experimental_make_numpy_dataset(self, numpy_input, session):
        return numpy_dataset.one_host_numpy_dataset(numpy_input, numpy_dataset.SingleDevice(self._host_device), session)

    def _get_input_workers(self, options):
        if not options or options.experimental_fetch_to_device:
            return input_lib.InputWorkers(tuple(self._device_input_worker_devices.items()))
        else:
            return input_lib.InputWorkers(tuple(self._host_input_worker_devices.items()))

    def _check_spec(self, element_spec):
        if isinstance(element_spec, values.PerReplicaSpec):
            element_spec = element_spec._component_specs
        specs = nest.flatten_with_joined_string_paths(element_spec)
        for path, spec in specs:
            if isinstance(spec, (sparse_tensor.SparseTensorSpec, ragged_tensor.RaggedTensorSpec)):
                raise ValueError('Found tensor {} with spec {}. TPUStrategy does not support distributed datasets with device prefetch when using sparse or ragged tensors. If you intend to use sparse or ragged tensors, please pass a tf.distribute.InputOptions object with experimental_fetch_to_device set to False to your dataset distribution function.'.format(path, type(spec)))

    def _experimental_distribute_dataset(self, dataset, options):
        if options and options.experimental_replication_mode == distribute_lib.InputReplicationMode.PER_REPLICA:
            raise NotImplementedError('InputReplicationMode.PER_REPLICA is only supported in `experimental_distribute_datasets_from_function`.')
        if options is None or options.experimental_fetch_to_device:
            self._check_spec(dataset.element_spec)
        return input_util.get_distributed_dataset(dataset, self._get_input_workers(options), self._container_strategy(), num_replicas_in_sync=self._num_replicas_in_sync, options=options, replica_order=self._replica_order)

    def _distribute_datasets_from_function(self, dataset_fn, options):
        if options and options.experimental_replication_mode == distribute_lib.InputReplicationMode.PER_REPLICA:
            raise NotImplementedError('InputReplicationMode.PER_REPLICA is only supported in  `experimental_distribute_datasets_from_function` of tf.distribute.MirroredStrategy')
        input_workers = self._get_input_workers(options)
        input_contexts = []
        num_workers = input_workers.num_workers
        for i in range(num_workers):
            input_contexts.append(distribute_lib.InputContext(num_input_pipelines=num_workers, input_pipeline_id=i, num_replicas_in_sync=self._num_replicas_in_sync))
        distributed_dataset = input_util.get_distributed_datasets_from_function(dataset_fn, input_workers, input_contexts, self._container_strategy(), options=options, replica_order=self._replica_order)
        if options is None or options.experimental_fetch_to_device:
            self._check_spec(distributed_dataset.element_spec)
        return distributed_dataset

    def _experimental_distribute_values_from_function(self, value_fn):
        per_replica_values = []
        for replica_id in range(self._num_replicas_in_sync):
            per_replica_values.append(value_fn(distribute_lib.ValueContext(replica_id, self._num_replicas_in_sync)))
        return distribute_utils.regroup(per_replica_values, always_wrap=True)

    def _experimental_run_steps_on_iterator(self, fn, multi_worker_iterator, iterations, initial_loop_values=None):
        if initial_loop_values is None:
            initial_loop_values = {}
        initial_loop_values = nest.flatten(initial_loop_values)
        ctx = input_lib.MultiStepContext()

        def run_fn(inputs):
            """Single step on the TPU device."""
            fn_result = fn(ctx, inputs)
            flat_last_step_outputs = nest.flatten(ctx.last_step_outputs)
            if flat_last_step_outputs:
                with ops.control_dependencies([fn_result]):
                    return [array_ops.identity(f) for f in flat_last_step_outputs]
            else:
                return fn_result
        self._outer_control_flow_context = ops.get_default_graph()._get_control_flow_context()

        def rewrite_fn(*args):
            """The rewritten step fn running on TPU."""
            del args
            per_replica_inputs = multi_worker_iterator.get_next()
            replicate_inputs = []
            for replica_id in range(self._num_replicas_in_sync):
                select_replica = lambda x: distribute_utils.select_replica(replica_id, x)
                replicate_inputs.append((nest.map_structure(select_replica, per_replica_inputs),))
            replicate_outputs = tpu.replicate(run_fn, replicate_inputs, device_assignment=self._device_assignment, xla_options=tpu.XLAOptions(use_spmd_for_xla_partitioning=self._use_spmd_for_xla_partitioning))
            if isinstance(replicate_outputs[0], list):
                replicate_outputs = nest.flatten(replicate_outputs)
            return replicate_outputs
        assert isinstance(initial_loop_values, list)
        initial_loop_values = initial_loop_values * self._num_replicas_in_sync
        with ops.device(self._host_device):
            if self.steps_per_run == 1:
                replicate_outputs = rewrite_fn()
            else:
                replicate_outputs = training_loop.repeat(iterations, rewrite_fn, initial_loop_values)
        del self._outer_control_flow_context
        ctx.run_op = control_flow_ops.group(replicate_outputs)
        if isinstance(replicate_outputs, list):
            last_step_tensor_outputs = [x for x in replicate_outputs if not isinstance(x, ops.Operation)]
            output_num = len(last_step_tensor_outputs) // self._num_replicas_in_sync
            last_step_tensor_outputs = [last_step_tensor_outputs[i::output_num] for i in range(output_num)]
        else:
            last_step_tensor_outputs = []
        _set_last_step_outputs(ctx, last_step_tensor_outputs)
        return ctx

    def _call_for_each_replica(self, fn, args, kwargs):
        with _TPUReplicaContext(self._container_strategy()):
            return fn(*args, **kwargs)

    @contextlib.contextmanager
    def experimental_logical_device(self, logical_device_id):
        """Places variables and ops on the specified logical device."""
        num_logical_devices_per_replica = self._tpu_devices.shape[1]
        if logical_device_id >= num_logical_devices_per_replica:
            raise ValueError('`logical_device_id` not in range (was {}, but there are only {} logical devices per replica).'.format(logical_device_id, num_logical_devices_per_replica))
        self._logical_device_stack.append(logical_device_id)
        try:
            if tpu_util.enclosing_tpu_context() is None:
                yield
            else:
                with ops.device(tpu.core(logical_device_id)):
                    yield
        finally:
            self._logical_device_stack.pop()

    def _experimental_initialize_system(self):
        """Experimental method added to be used by Estimator.

    This is a private method only to be used by Estimator. Other frameworks
    should directly be calling `tf.tpu.experimental.initialize_tpu_system`
    """
        tpu_cluster_resolver_lib.initialize_tpu_system(self._tpu_cluster_resolver)

    def _create_variable(self, next_creator, **kwargs):
        """Create a TPUMirroredVariable. See `DistributionStrategy.scope`."""
        if kwargs.pop('skip_mirrored_creator', False):
            return next_creator(**kwargs)
        custom_tpu_variable_creator = kwargs.pop('custom_tpu_variable_creator', None)
        if custom_tpu_variable_creator is not None:
            return custom_tpu_variable_creator(next_creator, **kwargs)
        colocate_with = kwargs.pop('colocate_with', None)
        if colocate_with is None:
            devices = self._tpu_devices[:, self._logical_device_stack[-1]]
        elif isinstance(colocate_with, numpy_dataset.SingleDevice):
            with ops.device(colocate_with.device):
                return next_creator(**kwargs)
        else:
            devices = colocate_with._devices
        num_replicas, num_cores_per_replica = self._tpu_devices.shape

        def _create_mirrored_tpu_variables(**kwargs):
            """Returns a list of `tf.Variable`s.

      The list contains `number_replicas` `tf.Variable`s and can be used to
      initialize a `TPUMirroredVariable`.

      Args:
        **kwargs: the keyword arguments for creating a variable
      """
            initial_value = None
            value_list = []
            for i, d in enumerate(devices):
                with ops.device(d):
                    if i == 0:
                        initial_value = kwargs['initial_value']
                        with maybe_init_scope():
                            initial_value = initial_value() if callable(initial_value) else initial_value
                    if i > 0:
                        var0name = value_list[0].name.split(':')[0]
                        kwargs['name'] = '%s/replica_%d/' % (var0name, i)
                    kwargs['initial_value'] = initial_value
                    with context.device_policy(context.DEVICE_PLACEMENT_SILENT):
                        v = next_creator(**kwargs)
                    assert not isinstance(v, tpu_values.TPUMirroredVariable)
                    value_list.append(v)
            return value_list

        def _create_mirrored_tpu_replicated_variables(**kwargs):
            """Returns a list of `TPUReplicatedVariable`s.

      The list consists of `num_replicas` `TPUReplicatedVariable`s and can be
      used to initialize a `TPUMirroredVariable`. Each `TPUReplicatedVariable`
      contains a list of `tf.Variable`s which are replicated to
      `num_cores_per_replica` logical cores to enable XLA SPMD compilation.

      Args:
        **kwargs: the keyword arguments for creating a variable
      """
            initial_value = kwargs['initial_value']
            with maybe_init_scope():
                initial_value = initial_value() if callable(initial_value) else initial_value
            mirrored_replicated_var_list = []
            for replica_id in range(num_replicas):
                replicated_var_list = []
                for logic_core_id in range(num_cores_per_replica):
                    with ops.device(self._tpu_devices[replica_id][logic_core_id]):
                        kwargs['initial_value'] = initial_value
                        v = next_creator(**kwargs)
                    replicated_var_list.append(v)
                replica_name = '{}/r:{}'.format(kwargs['name'], replica_id)
                tpu_replicated_var = tpu_replicated_variable.TPUReplicatedVariable(variables=replicated_var_list, name=replica_name)
                mirrored_replicated_var_list.append(tpu_replicated_var)
            return mirrored_replicated_var_list

        def uninitialized_variable_creator(**kwargs):
            uninitialized_variable = tpu_util.TPUUninitializedVariable(**kwargs)
            self.lazy_variable_tracker.add_uninitialized_var(uninitialized_variable)
            setattr(uninitialized_variable, '_lazy_scope', self.lazy_variable_tracker)
            return uninitialized_variable

        def _create_uninitialized_mirrored_tpu_variables(**kwargs):
            """Returns a list of `tf.Variable`s.

      The list contains `number_replicas` `tf.Variable`s and can be used to
      initialize a `TPUMirroredVariable`.

      Args:
        **kwargs: the keyword arguments for creating a variable
      """
            if kwargs.get('initial_value', None) is None:
                return _create_mirrored_tpu_variables(**kwargs)
            value_list = []
            for i, d in enumerate(devices):
                with ops.device(d):
                    if i == 0:
                        initial_value = kwargs.get('initial_value', None)
                        with maybe_init_scope():
                            if initial_value is not None:
                                if callable(initial_value):
                                    initial_value = initial_value()
                                initial_value = ops.convert_to_tensor(initial_value, dtype=kwargs.get('dtype', None))
                    if i > 0:
                        var0name = value_list[0].name.split(':')[0]
                        kwargs['name'] = '%s/replica_%d/' % (var0name, i)
                    kwargs['initial_value'] = initial_value
                    if kwargs.get('dtype', None) is None:
                        kwargs['dtype'] = kwargs['initial_value'].dtype
                    if kwargs.get('shape', None) is None:
                        kwargs['shape'] = kwargs['initial_value'].shape
                    with context.device_policy(context.DEVICE_PLACEMENT_SILENT):
                        v = uninitialized_variable_creator(**kwargs)
                    assert not isinstance(v, tpu_values.TPUMirroredVariable)
                    value_list.append(v)
            return value_list

        def _create_uninitialized_mirrored_tpu_replicated_variables(**kwargs):
            """Returns a list of `TPUReplicatedVariable`s.

      The list consists of `num_replicas` `TPUReplicatedVariable`s and can be
      used to initialize a `TPUMirroredVariable`. Each `TPUReplicatedVariable`
      contains a list of `tf.Variable`s which are replicated to
      `num_cores_per_replica` logical cores to enable XLA SPMD compilation.

      Args:
        **kwargs: the keyword arguments for creating a variable
      """
            dtype = kwargs.get('dtype', None)
            shape = kwargs.get('shape', None)
            initial_value = kwargs.get('initial_value', None)
            if initial_value is None:
                return _create_mirrored_tpu_replicated_variables(**kwargs)
            with maybe_init_scope():
                if initial_value is not None:
                    if callable(initial_value):
                        initial_value = initial_value()
                    initial_value = ops.convert_to_tensor(initial_value, dtype=dtype)
                    kwargs['initial_value'] = initial_value
                    if dtype is None:
                        kwargs['dtype'] = kwargs['initial_value'].dtype
                    if shape is None:
                        kwargs['shape'] = kwargs['initial_value'].shape
            mirrored_replicated_var_list = []
            for replica_id in range(num_replicas):
                replicated_var_list = []
                for logic_core_id in range(num_cores_per_replica):
                    with ops.device(self._tpu_devices[replica_id][logic_core_id]):
                        v = uninitialized_variable_creator(**kwargs)
                    replicated_var_list.append(v)
                replica_name = '{}/r:{}'.format(kwargs['name'], replica_id)
                tpu_replicated_var = tpu_replicated_variable.TPUReplicatedVariable(variables=replicated_var_list, name=replica_name)
                mirrored_replicated_var_list.append(tpu_replicated_var)
            return mirrored_replicated_var_list
        if not self._using_custom_device and enable_batch_variable_initialization():
            if self._use_spmd_for_xla_partitioning and num_cores_per_replica > 1:
                real_creator = _create_uninitialized_mirrored_tpu_replicated_variables
            else:
                real_creator = _create_uninitialized_mirrored_tpu_variables
            kwargs['experimental_batch_initialization'] = True
        elif self._use_spmd_for_xla_partitioning and num_cores_per_replica > 1:
            real_creator = _create_mirrored_tpu_replicated_variables
        else:
            real_creator = _create_mirrored_tpu_variables
        mirrored_variable = distribute_utils.create_mirrored_variable(self._container_strategy(), real_creator, distribute_utils.TPU_VARIABLE_CLASS_MAPPING, distribute_utils.TPU_VARIABLE_POLICY_MAPPING, **kwargs)
        if not self._using_custom_device and enable_batch_variable_initialization():
            setattr(mirrored_variable, '_lazy_scope', self.lazy_variable_tracker)
        return mirrored_variable

    @property
    def lazy_variable_tracker(self):
        if not getattr(self, '_lazy_variable_tracker', None):
            self._lazy_variable_tracker = tpu_util.LazyVariableTracker()
        return self._lazy_variable_tracker

    def _resource_creator_scope(self):

        def lookup_creator(next_creator, *args, **kwargs):
            host_to_table = collections.OrderedDict()
            for host_device in self._device_input_worker_devices.keys():
                with ops.device(host_device):
                    host_to_table[host_device] = next_creator(*args, **kwargs)
            return values.PerWorkerResource(self._container_strategy(), host_to_table)
        return ops.resource_creator_scope('StaticHashTable', lookup_creator)

    def _gather_to_implementation(self, value, destinations, axis, options):
        if not isinstance(value, values.DistributedValues):
            return value
        value_list = list(value.values)
        if isinstance(value, values.DistributedVariable) and value._packed_variable is not None:
            value_list = list((value._packed_variable.on_device(d) for d in value._packed_variable.devices))
        if len(value.values) <= _XLA_OP_BY_OP_INPUTS_LIMIT:
            output = array_ops.concat(value_list, axis=axis)
        else:
            output = array_ops.concat(value_list[:_XLA_OP_BY_OP_INPUTS_LIMIT], axis=axis)
            for i in range(_XLA_OP_BY_OP_INPUTS_LIMIT, len(value_list), _XLA_OP_BY_OP_INPUTS_LIMIT - 1):
                output = array_ops.concat([output] + value_list[i:i + _XLA_OP_BY_OP_INPUTS_LIMIT - 1], axis=axis)
        output = self._broadcast_output(destinations, output)
        return output

    def _broadcast_output(self, destinations, output):
        devices = cross_device_ops_lib.get_devices_from(destinations)
        if len(devices) == 1:
            dest_canonical = device_util.canonicalize(devices[0])
            host_canonical = device_util.canonicalize(self._host_device)
            if dest_canonical != host_canonical:
                with ops.device(dest_canonical):
                    output = array_ops.identity(output)
        else:
            output = cross_device_ops_lib.simple_broadcast(output, destinations)
        return output

    def _reduce_to(self, reduce_op, value, destinations, options):
        if (isinstance(value, values.DistributedValues) or tensor_util.is_tf_type(value)) and tpu_util.enclosing_tpu_context() is not None:
            if reduce_op == reduce_util.ReduceOp.MEAN:
                value = math_ops.scalar_mul(1.0 / self._num_replicas_in_sync, value)
            elif reduce_op != reduce_util.ReduceOp.SUM:
                raise NotImplementedError(f'`reduce_op`={reduce_op} is not supported. Currently we only support ReduceOp.SUM and ReduceOp.MEAN in TPUStrategy.')
            return tpu_ops.cross_replica_sum(value)
        if not isinstance(value, values.DistributedValues):
            return cross_device_ops_lib.reduce_non_distributed_value(reduce_op, value, destinations, self._num_replicas_in_sync)
        value_list = value.values
        if isinstance(value, values.DistributedVariable) and value._packed_variable is not None:
            value_list = tuple((value._packed_variable.on_device(d) for d in value._packed_variable.devices))
        if len(value.values) <= _XLA_OP_BY_OP_INPUTS_LIMIT:
            output = math_ops.add_n(value_list)
        else:
            output = array_ops.zeros_like(value_list[0], dtype=value_list[0].dtype)
            for i in range(0, len(value_list), _XLA_OP_BY_OP_INPUTS_LIMIT):
                output += math_ops.add_n(value_list[i:i + _XLA_OP_BY_OP_INPUTS_LIMIT])
        if reduce_op == reduce_util.ReduceOp.MEAN:
            output *= 1.0 / len(value_list)
        output = self._broadcast_output(destinations, output)
        return output

    def _update(self, var, fn, args, kwargs, group):
        assert isinstance(var, tpu_values.TPUVariableMixin) or isinstance(var, resource_variable_ops.BaseResourceVariable)
        if tpu_util.enclosing_tpu_context() is not None:
            if group:
                return fn(var, *args, **kwargs)
            else:
                return (fn(var, *args, **kwargs),)
        packed_var = var._packed_variable
        if packed_var is not None and (not context.executing_eagerly()):
            if group:
                return fn(packed_var, *args, **kwargs)
            else:
                return (fn(packed_var, *args, **kwargs),)
        updates = []
        values_and_devices = []
        if packed_var is not None:
            for device in packed_var.devices:
                values_and_devices.append((packed_var, device))
        else:
            for value in var.values:
                values_and_devices.append((value, value.device))
        if var.synchronization != variables_lib.VariableSynchronization.ON_READ and var.aggregation != variables_lib.VariableAggregation.NONE:
            distribute_utils.assert_mirrored(args)
            distribute_utils.assert_mirrored(kwargs)
        for i, value_and_device in enumerate(values_and_devices):
            value = value_and_device[0]
            device = value_and_device[1]
            name = 'update_%d' % i
            with ops.device(device), distribute_lib.UpdateContext(i), ops.name_scope(name):
                updates.append(fn(value, *distribute_utils.select_replica(i, args), **distribute_utils.select_replica(i, kwargs)))
        return distribute_utils.update_regroup(self, updates, group)

    def read_var(self, var):
        assert isinstance(var, tpu_values.TPUVariableMixin) or isinstance(var, resource_variable_ops.BaseResourceVariable)
        return var.read_value()

    def value_container(self, value):
        return value

    def _broadcast_to(self, tensor, destinations):
        del destinations
        if isinstance(tensor, (float, int)):
            return tensor
        if tpu_util.enclosing_tpu_context() is not None:
            broadcast_tensor = [tensor for _ in range(self._num_replicas_in_sync)]
            result = tpu_ops.all_to_all(broadcast_tensor, concat_dimension=0, split_dimension=0, split_count=self._num_replicas_in_sync)
            return result[0]
        return tensor

    @property
    def num_hosts(self):
        if self._device_assignment is None:
            return self._tpu_metadata.num_hosts
        return len(set([self._device_assignment.host_device(r) for r in range(self._device_assignment.num_replicas)]))

    @property
    def num_replicas_per_host(self):
        if self._device_assignment is None:
            return self._tpu_metadata.num_of_cores_per_host
        max_models_per_host = self._tpu_metadata.num_of_cores_per_host // self._device_assignment.num_cores_per_replica
        return min(self._device_assignment.num_replicas, max_models_per_host)

    @property
    def _num_replicas_in_sync(self):
        if self._device_assignment is None:
            return self._tpu_metadata.num_cores
        return self._device_assignment.num_replicas

    @property
    def experimental_between_graph(self):
        return False

    @property
    def experimental_should_init(self):
        return True

    @property
    def should_checkpoint(self):
        return True

    @property
    def should_save_summary(self):
        return True

    @property
    def worker_devices(self):
        return tuple(self._tpu_devices[:, self._logical_device_stack[-1]])

    @property
    def parameter_devices(self):
        return self.worker_devices

    @property
    def tpu_hardware_feature(self):
        """Return the `tf.tpu.experimental.HardwareFeature` class."""
        return tpu_hardware_feature.HardwareFeature(self._tpu_cluster_resolver.tpu_hardware_feature)

    def non_slot_devices(self, var_list):
        return self._host_device

    def _update_non_slot(self, colocate_with, fn, args, kwargs, group):
        del colocate_with
        with ops.device(self._host_device), distribute_lib.UpdateContext(None):
            result = fn(*args, **kwargs)
            if group:
                return result
            else:
                return nest.map_structure(self._local_results, result)

    def _configure(self, session_config=None, cluster_spec=None, task_type=None, task_id=None):
        del cluster_spec, task_type, task_id
        if session_config:
            session_config.CopyFrom(self._update_config_proto(session_config))

    def _update_config_proto(self, config_proto):
        updated_config = copy.deepcopy(config_proto)
        updated_config.isolate_session_state = True
        cluster_spec = self._tpu_cluster_resolver.cluster_spec()
        if cluster_spec:
            updated_config.cluster_def.CopyFrom(cluster_spec.as_cluster_def())
        return updated_config

    @property
    def _global_batch_size(self):
        """`make_dataset_iterator` and `make_numpy_iterator` use global batch size.

    `make_input_fn_iterator` assumes per-replica batching.

    Returns:
      Boolean.
    """
        return True

    def tpu_run(self, fn, args, kwargs, options=None):
        func = self._tpu_function_creator(fn, options)
        return func(args, kwargs)

    def _tpu_function_creator(self, fn, options):
        if context.executing_eagerly() and fn in self._tpu_function_cache:
            return self._tpu_function_cache[fn]
        strategy = self._container_strategy()

        def tpu_function(args, kwargs):
            """TF Function used to replicate the user computation."""
            logging.vlog(1, '`TPUStrategy.run` is called with [args: %s] [kwargs: %s]', args, kwargs)
            if kwargs is None:
                kwargs = {}
            result = [[]]

            def replicated_fn(replica_id, replica_args, replica_kwargs):
                """Wraps user function to provide replica ID and `Tensor` inputs."""
                with _TPUReplicaContext(strategy, replica_id_in_sync_group=replica_id):
                    result[0] = fn(*replica_args, **replica_kwargs)
                return result[0]
            replicate_inputs = []
            for i in range(strategy.num_replicas_in_sync):
                replicate_inputs.append([constant_op.constant(i, dtype=dtypes.int32), distribute_utils.select_replica(i, args), distribute_utils.select_replica(i, kwargs)])
            if options.experimental_enable_dynamic_batch_size and replicate_inputs:
                maximum_shapes = []
                flattened_list = nest.flatten(replicate_inputs[0])
                for input_tensor in flattened_list:
                    if tensor_util.is_tf_type(input_tensor):
                        rank = input_tensor.shape.rank
                    else:
                        rank = np.ndim(input_tensor)
                    if rank is None:
                        raise ValueError('input tensor {} to TPUStrategy.run() has unknown rank, which is not allowed'.format(input_tensor))
                    maximum_shape = tensor_shape.TensorShape([None] * rank)
                    maximum_shapes.append(maximum_shape)
                maximum_shapes = nest.pack_sequence_as(replicate_inputs[0], maximum_shapes)
            else:
                maximum_shapes = None
            if options.experimental_bucketizing_dynamic_shape:
                padding_spec = tpu.PaddingSpec.POWER_OF_TWO
            else:
                padding_spec = None
            with strategy.scope():
                xla_options = options.experimental_xla_options or tpu.XLAOptions(use_spmd_for_xla_partitioning=self._use_spmd_for_xla_partitioning)
                replicate_outputs = tpu.replicate(replicated_fn, replicate_inputs, device_assignment=self._device_assignment, maximum_shapes=maximum_shapes, padding_spec=padding_spec, xla_options=xla_options)
            filter_ops = lambda x: [o for o in x if not isinstance(o, ops.Operation)]
            if isinstance(result[0], list):
                result[0] = filter_ops(result[0])
            if result[0] is None or isinstance(result[0], ops.Operation):
                replicate_outputs = [None] * len(replicate_outputs)
            else:
                replicate_outputs = [nest.pack_sequence_as(result[0], filter_ops(nest.flatten(output))) for output in replicate_outputs]
            return distribute_utils.regroup(replicate_outputs)
        if context.executing_eagerly():
            tpu_function = def_function.function(tpu_function)
            self._tpu_function_cache[fn] = tpu_function
        return tpu_function

    def _in_multi_worker_mode(self):
        """Whether this strategy indicates working in multi-worker settings."""
        return False

    def _get_local_replica_id(self, replica_id_in_sync_group):
        return replica_id_in_sync_group