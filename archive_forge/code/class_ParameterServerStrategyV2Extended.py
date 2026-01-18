import functools
import os
import threading
from tensorflow.python.distribute import cross_device_ops as cross_device_ops_lib
from tensorflow.python.distribute import device_util
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import input_lib
from tensorflow.python.distribute import input_util
from tensorflow.python.distribute import mirrored_run
from tensorflow.python.distribute import multi_worker_util
from tensorflow.python.distribute import parameter_server_strategy
from tensorflow.python.distribute import ps_values
from tensorflow.python.distribute import sharded_variable
from tensorflow.python.distribute import values
from tensorflow.python.distribute.cluster_resolver import cluster_resolver as base_cluster_resolver
from tensorflow.python.distribute.coordinator import cluster_coordinator
from tensorflow.python.eager import context
from tensorflow.python.eager import remote
from tensorflow.python.framework import config
from tensorflow.python.framework import device as tf_device
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.trackable import base as trackable
from tensorflow.python.training import server_lib
from tensorflow.python.util import keras_deps
from tensorflow.python.util import nest
from tensorflow.python.util import tf_inspect
from tensorflow.python.util.tf_export import tf_export
from tensorflow.tsl.protobuf import coordination_config_pb2
class ParameterServerStrategyV2Extended(parameter_server_strategy.ParameterServerStrategyExtended):
    """Extended class for ParameterServerStrategyV2.

  Please see `tf.distribute.StrategyExtended` doc for more information.
  """

    def __init__(self, container_strategy, cluster_resolver: base_cluster_resolver.ClusterResolver, variable_partitioner):
        """Initialization of ParameterServerStrategyV2Extended."""
        super(ParameterServerStrategyV2Extended, self).__init__(container_strategy)
        self._num_ps = len(cluster_resolver.cluster_spec().as_dict().get('ps', []))
        self._num_workers = len(cluster_resolver.cluster_spec().as_dict().get('worker', []))
        self._variable_count = 0
        self._variable_partitioner = variable_partitioner
        self._used_with_coordinator = False
        self._being_scheduled = False
        self._set_num_gpus()
        distribute_lib.distribution_strategy_replica_gauge.get_cell('num_gpus_per_worker').set(self._num_gpus_per_worker)
        self._cross_device_ops = cross_device_ops_lib.ReductionToOneDevice(reduce_to_device='/device:CPU:0')
        self._cross_device_ops._canonicalize_devices = False
        self._allow_run_without_coordinator = False
        self._coordinator_creation_lock = threading.Lock()

    def _set_num_gpus(self):
        devices = config.list_logical_devices('GPU')
        per_worker_gpus = {}
        for d in devices:
            d_spec = tf_device.DeviceSpec.from_string(d.name)
            if d_spec.device_type == 'GPU' and d_spec.job == 'worker':
                job_spec = d_spec.replace(device_type=None, device_index=None)
                per_worker_gpus[job_spec] = per_worker_gpus.get(job_spec, 0) + 1
        num_gpus = 0
        for _, count in per_worker_gpus.items():
            if num_gpus > 0 and count != num_gpus:
                raise ValueError('Mismatched number of GPUs per worker')
            num_gpus = count
        self._num_gpus_per_worker = num_gpus
        logging.info(f'Number of GPUs on workers: {self._num_gpus_per_worker}')

    @property
    def _num_replicas_in_sync(self):
        return self._num_gpus_per_worker or 1

    def _create_var_creator(self, next_creator, **kwargs):
        aggregation = kwargs.pop('aggregation', vs.VariableAggregation.NONE)

        def var_creator(**kwargs):
            """Create an AggregatingVariable."""
            v = next_creator(**kwargs)
            wrapped_v = ps_values.CachingVariable(v)
            wrapped = ps_values.AggregatingVariable(self._container_strategy(), wrapped_v, aggregation)
            return wrapped
        if self._num_replicas_in_sync > 1:
            if aggregation not in (vs.VariableAggregation.NONE, vs.VariableAggregation.SUM, vs.VariableAggregation.MEAN, vs.VariableAggregation.ONLY_FIRST_REPLICA):
                raise ValueError('Invalid variable aggregation mode: ' + aggregation + ' for variable: ' + kwargs['name'])
            return var_creator
        else:

            def variable_creator_single_replica(**kwargs):
                v = next_creator(**kwargs)
                return ps_values.CachingVariable(v)
            return variable_creator_single_replica

    def _create_per_worker_variable(self, next_creator, **kwargs):
        """Create an unsynced, unaggregated variable on each worker."""
        return ps_values.PerWorkerVariable(self._container_strategy(), next_creator, **kwargs)

    def _create_variable(self, next_creator, **kwargs):
        """Implements StrategyExtendedV2._create_variable.

    Creates a `Variable` or a `ShardedVariable`. A `ShardedVariable` will be
    created if satisfying all the following criteria:
      1. `self._variable_partitioner` results in more than one partition on the
         first axis.
      2. variable's rank is greater than 0.
      3. variable is not colocated with another variable.
    Otherwise a `Variable` will be created.

    Args:
      next_creator: See `variable_scope.variable_creator_scope`; the next
        creator in the chain.
      **kwargs: Passed through to the next creator.

    Returns:
      A `Variable` or `ShardedVariable`.
    """
        if kwargs.pop('per_worker_variable', False):
            logging.info('Creating per worker variable')
            return self._create_per_worker_variable(next_creator, **kwargs)
        var_creator = self._create_var_creator(next_creator, **kwargs)
        if 'colocate_with' in kwargs:
            colocate_with = kwargs['colocate_with']
            with ops.device(None):
                with ops.colocate_with(colocate_with):
                    var = var_creator(**kwargs)
                    logging.debug('Creating variable (name:%s, shape:%r) that colocates with %s', var.name, var.shape, kwargs['colocate_with'].name)
                    return var
        if self._variable_partitioner is None:
            return self._create_variable_round_robin(var_creator, **kwargs)
        name = kwargs.get('name', None)
        dtype = kwargs.get('dtype', None)
        shape = kwargs.get('shape', None)
        initial_value = kwargs.get('initial_value', None)
        if initial_value is None:
            v = next_creator(**kwargs)
            if not isinstance(v, resource_variable_ops.UninitializedVariable):
                raise ValueError('It looks like you are using `ParameterServerStrategy` with a `variable_partitioner`, and trying to create a variable without specifying `initial_value`. This is not allowed. Please specify the `initial_value`.')
            elif shape is None or dtype is None:
                raise ValueError('It looks like you are trying to load a `SavedModel` using `tf.saved_model.load` within a `ParameterServerStrategy` scope, but the `SavedModel` is missing shape or dtype information.')
            else:

                def initializer(shape, dtype, **kwargs):
                    if 'partition_shape' in kwargs:
                        shape = kwargs['partition_shape']
                    return array_ops.zeros(shape, dtype)
                initial_value = functools.partial(initializer, shape=shape, dtype=dtype)
        init_from_fn = callable(initial_value)
        if init_from_fn and (shape is None or dtype is None):
            init_from_fn = False
            initial_value = initial_value()
        if not init_from_fn:
            initial_value = ops.convert_to_tensor(initial_value, dtype=dtype)
            dtype = initial_value.dtype
            shape = initial_value.shape
        else:
            shape = tensor_shape.as_shape(shape)
        if shape.rank == 0:
            return self._create_variable_round_robin(var_creator, **kwargs)
        num_partitions = self._variable_partitioner(shape=shape, dtype=dtype)
        if not num_partitions or num_partitions[0] == 0 or any((v != 1 for v in num_partitions[1:])):
            raise ValueError('variable_partitioner must return a list/tuple whose elements are 1 besides the first element (non-zero), got: %r' % num_partitions)
        if num_partitions[0] == 1:
            return self._create_variable_round_robin(var_creator, **kwargs)
        num_partitions = min(num_partitions[0], shape[0])
        base = shape[0] // num_partitions
        extra = shape[0] % num_partitions
        offsets = []
        for i in range(num_partitions):
            if i == 0:
                offsets.append(0)
            else:
                prev_shard_size = base + (1 if i - 1 < extra else 0)
                offsets.append(offsets[i - 1] + prev_shard_size)
        offsets.append(shape[0])

        def init_shard_fn(shard_index):
            if not init_from_fn:
                logging.log_if(logging.WARN, _INEFFICIENT_INIT_WARNING % name, shard_index == 0 and shape.num_elements() > _LARGE_VARIABLE_NUM_ELEMENTS)
                return initial_value[offsets[shard_index]:offsets[shard_index + 1]]
            partition_shape = (offsets[shard_index + 1] - offsets[shard_index],) + shape[1:]
            partition_offset = (offsets[shard_index],) + (0,) * len(shape[1:])
            arg_spec = tf_inspect.getfullargspec(initial_value)
            if 'shard_info' not in arg_spec.args and 'shard_info' not in arg_spec.kwonlyargs:
                try:
                    value = initial_value(partition_shape=partition_shape, partition_offset=partition_offset)
                except (TypeError, ValueError):
                    value = initial_value()
                if value.shape == partition_shape:
                    return value
                else:
                    logging.log_if(logging.WARN, _INEFFICIENT_INIT_WARNING % name, shard_index == 0 and shape.num_elements() > _LARGE_VARIABLE_NUM_ELEMENTS)
                    return value[offsets[shard_index]:offsets[shard_index + 1]]
            else:
                return initial_value(shard_info=trackable.ShardInfo(shape=tensor_shape.as_shape(partition_shape), offset=partition_offset))
        var_list = []
        for i in range(num_partitions):
            kwargs['shape'] = (offsets[i + 1] - offsets[i],) + shape[1:]
            kwargs['initial_value'] = lambda: init_shard_fn(i)
            if name is not None:
                kwargs['name'] = '{}/part_{}'.format(name, i)
            var_list.append(self._create_variable_round_robin(var_creator, **kwargs))
        result = sharded_variable.ShardedVariable(var_list)
        return result

    def _create_variable_round_robin(self, next_creator, **kwargs):
        with ops.colocate_with(None, ignore_existing=True):
            with ops.device('/job:ps/task:%d/device:CPU:0' % (self._variable_count % self._num_ps)):
                var = next_creator(**kwargs)
                log_method = logging.info if os.getenv('TF_PSS_VERBOSE_VARIABLE_PLACEMENT') else logging.debug
                log_method('Creating variable (name:%s, shape:%r) on /job:ps/task:%d/device:CPU:0', var.name, var.shape, self._variable_count % self._num_ps)
                self._variable_count += 1
                return var

    def _resource_creator_scope(self):
        with self._coordinator_creation_lock:
            if not self._container_strategy()._cluster_coordinator:
                cluster_coordinator.ClusterCoordinator(strategy=self._container_strategy())

        def lookup_creator(next_creator, *args, **kwargs):
            if keras_deps.get_load_context_function()():
                return ps_values.RestoredDistributedTable(self._container_strategy(), lambda: next_creator(*args, **kwargs))
            else:
                return ps_values.DistributedTable(self._container_strategy(), lambda: next_creator(*args, **kwargs))

        def restored_lookup_creator(next_creator, *args, **kwargs):
            return ps_values.RestoredDistributedTable(self._container_strategy(), lambda: next_creator(*args, **kwargs))
        return [ops.resource_creator_scope('StaticHashTable', lookup_creator), ops.resource_creator_scope('RestoredStaticHashTable', restored_lookup_creator)]

    def _assert_used_with_cluster_coordinator(self):
        if not self._used_with_coordinator and (not self._allow_run_without_coordinator):
            raise NotImplementedError('`tf.distribute.experimental.ParameterServerStrategy` must be used with `tf.distribute.experimental.coordinator.ClusterCoordinator` in a custom training loop. If you are using `Model.fit`, please supply a dataset function directly to a `tf.keras.utils.experimental.DatasetCreator` instead.')

    def _assert_being_scheduled_by_cluster_coordinator(self):
        if not self._being_scheduled and (not self._allow_run_without_coordinator):
            logging.warning('A `tf.distribute.experimental.ParameterServerStrategy` method is invoked without using `ClusterCoordinator.schedule`. If you are not tracing a tf.function, this method is possibly executed on the coordinator, which can be slow. To properly dispatch functions to run on workers, methods like `run` or `reduce` should be used within a function passed to `tf.distribute.experimental.coordinator.ClusterCoordinator.schedule`.')

    def _input_workers_with_options(self, options=None):
        input_workers_devices = (('/device:CPU:0', self.worker_devices),)
        return input_lib.InputWorkers(input_workers_devices, canonicalize_devices=False)

    def _experimental_distribute_dataset(self, dataset, options):
        input_workers_devices = self._input_workers_with_options()
        return input_util.get_distributed_dataset(dataset, input_workers_devices, self._container_strategy(), num_replicas_in_sync=self._num_replicas_in_sync, options=options, build=ops.inside_function())

    def _distribute_datasets_from_function(self, dataset_fn, options):
        input_pipeline_id_in_sync = 0
        num_input_pipelines_in_sync = 1
        input_context = distribute_lib.InputContext(num_input_pipelines=num_input_pipelines_in_sync, input_pipeline_id=input_pipeline_id_in_sync, num_replicas_in_sync=self._num_replicas_in_sync)
        return input_util.get_distributed_datasets_from_function(dataset_fn, self._input_workers_with_options(options), [input_context], self._container_strategy(), options=options, build=ops.inside_function())

    @property
    def worker_devices(self):
        num_gpus = self._num_gpus_per_worker
        if num_gpus > 0:
            compute_devices = tuple(('/device:GPU:%d' % (i,) for i in range(num_gpus)))
        else:
            compute_devices = ('/device:CPU:0',)
        return compute_devices

    def _call_for_each_replica(self, fn, args, kwargs):
        self._assert_being_scheduled_by_cluster_coordinator()
        return mirrored_run.call_for_each_replica(self._container_strategy(), fn, args, kwargs)

    def _reduce(self, reduce_op, value):
        self._assert_being_scheduled_by_cluster_coordinator()
        dst = device_util.current() or self._default_device or '/device:CPU:0'
        destinations = device_util.canonicalize_without_job_and_task(dst)
        result = self._local_results(self.reduce_to(reduce_op, value, destinations))[0]
        return result

    def _reduce_to(self, reduce_op, value, destinations, options):
        self._assert_being_scheduled_by_cluster_coordinator()

        def get_values(x):
            if isinstance(x, values.DistributedValues):
                return self._cross_device_ops.reduce(reduce_op, x, destinations=destinations)
            return x
        return nest.map_structure(get_values, value)