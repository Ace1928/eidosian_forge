from tensorflow.python.distribute import device_util
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import distribute_utils
from tensorflow.python.distribute import input_lib
from tensorflow.python.distribute import input_util
from tensorflow.python.distribute import numpy_dataset
from tensorflow.python.distribute.v1 import input_lib as input_lib_v1
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import while_loop
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import tf_export
class OneDeviceExtended(distribute_lib.StrategyExtendedV1):
    """Implementation of OneDeviceStrategy."""

    def __init__(self, container_strategy, device):
        super(OneDeviceExtended, self).__init__(container_strategy)
        self._device = device_util.resolve(device)
        self._input_device = device_util.get_host_for_device(self._device)

    def _input_workers_with_options(self, options=None):
        if not options or options.experimental_fetch_to_device:
            return input_lib.InputWorkers([(self._input_device, (self._device,))])
        else:
            return input_lib.InputWorkers([(self._input_device, (self._input_device,))])

    @property
    def _input_workers(self):
        return self._input_workers_with_options()

    def _create_variable(self, next_creator, **kwargs):
        colocate_with = kwargs.pop('colocate_with', None)
        if colocate_with is None:
            with ops.device(self._device):
                return next_creator(**kwargs)
        elif isinstance(colocate_with, numpy_dataset.SingleDevice):
            with ops.device(colocate_with.device):
                return next_creator(**kwargs)
        else:
            with ops.colocate_with(colocate_with):
                return next_creator(**kwargs)

    def _validate_colocate_with_variable(self, colocate_with_variable):
        distribute_utils.validate_colocate(colocate_with_variable, self)

    def _make_dataset_iterator(self, dataset):
        """Make iterator from dataset without splitting the batch."""
        return input_lib_v1.DatasetIterator(dataset, self._input_workers, self._container_strategy())

    def _make_input_fn_iterator(self, input_fn, replication_mode=distribute_lib.InputReplicationMode.PER_WORKER):
        return input_lib_v1.InputFunctionIterator(input_fn, self._input_workers, [distribute_lib.InputContext()], self._container_strategy())

    def _experimental_make_numpy_dataset(self, numpy_input, session):
        return numpy_dataset.one_host_numpy_dataset(numpy_input, numpy_dataset.SingleDevice(self._input_device), session)

    def _broadcast_to(self, tensor, destinations):
        del destinations
        return tensor

    def _experimental_distribute_dataset(self, dataset, options):
        if options and options.experimental_replication_mode == distribute_lib.InputReplicationMode.PER_REPLICA:
            raise NotImplementedError('InputReplicationMode.PER_REPLICA is only supported in  `experimental_distribute_datasets_from_function`.')
        return input_util.get_distributed_dataset(dataset, self._input_workers_with_options(options), self._container_strategy(), options=options)

    def _distribute_datasets_from_function(self, dataset_fn, options):
        if options and options.experimental_replication_mode == distribute_lib.InputReplicationMode.PER_REPLICA:
            raise NotImplementedError('InputReplicationMode.PER_REPLICA is only supported in `experimental_distribute_datasets_from_function` of tf.distribute.MirroredStrategy')
        return input_util.get_distributed_datasets_from_function(dataset_fn, self._input_workers_with_options(options), [distribute_lib.InputContext()], self._container_strategy(), options=options)

    def _experimental_distribute_values_from_function(self, value_fn):
        return value_fn(distribute_lib.ValueContext())

    def _experimental_run_steps_on_iterator(self, fn, iterator, iterations, initial_loop_values=None):
        if initial_loop_values is None:
            initial_loop_values = {}
        initial_loop_values = nest.flatten(initial_loop_values)
        ctx = input_lib.MultiStepContext()

        def body(i, *args):
            """A wrapper around `fn` to create the while loop body."""
            del args
            fn_result = fn(ctx, iterator.get_next())
            flat_last_step_outputs = nest.flatten(ctx.last_step_outputs)
            with ops.control_dependencies([fn_result]):
                return [i + 1] + flat_last_step_outputs
        self._outer_control_flow_context = ops.get_default_graph()._get_control_flow_context()
        cond = lambda i, *args: i < iterations
        i = constant_op.constant(0)
        loop_result = while_loop.while_loop(cond, body, [i] + initial_loop_values, name='', parallel_iterations=1, back_prop=False, swap_memory=False, return_same_structure=True)
        del self._outer_control_flow_context
        ctx.run_op = control_flow_ops.group(loop_result)
        last_step_tensor_outputs = loop_result[1:]
        last_step_tensor_outputs_dict = nest.pack_sequence_as(ctx.last_step_outputs, last_step_tensor_outputs)
        ctx._set_last_step_outputs(last_step_tensor_outputs_dict)
        return ctx

    def _call_for_each_replica(self, fn, args, kwargs):
        strategy = self._container_strategy()
        with ops.device(self._device), _OneDeviceReplicaContext(strategy):
            return fn(*args, **kwargs)

    def _reduce_to(self, reduce_op, value, destinations, options):
        del reduce_op, destinations, options
        return value

    def _gather_to_implementation(self, value, destinations, axis, options):
        del destinations, axis, options
        return value

    def _update(self, var, fn, args, kwargs, group):
        return self._update_non_slot(var, fn, (var,) + tuple(args), kwargs, group)

    def _update_non_slot(self, colocate_with, fn, args, kwargs, group):
        del colocate_with
        with ops.device(self._device), distribute_lib.UpdateContext(self._device):
            result = fn(*args, **kwargs)
            if group:
                return result
            else:
                return nest.map_structure(self._local_results, result)

    def read_var(self, replica_local_var):
        """Read the aggregate value of a replica-local variable."""
        return array_ops.identity(replica_local_var)

    def _local_results(self, value):
        return (value,)

    def value_container(self, value):
        return value

    def _in_multi_worker_mode(self):
        """Whether this strategy indicates working in multi-worker settings."""
        return False

    @property
    def _num_replicas_in_sync(self):
        return 1

    @property
    def worker_devices(self):
        return (self._device,)

    @property
    def parameter_devices(self):
        return (self._device,)

    def non_slot_devices(self, var_list):
        del var_list
        return (self._device,)

    @property
    def experimental_should_init(self):
        return True

    @property
    def experimental_between_graph(self):
        return False

    @property
    def should_checkpoint(self):
        return True

    @property
    def should_save_summary(self):
        return True

    @property
    def _global_batch_size(self):
        """Global and per-replica batching are equivalent for OneDeviceStrategy."""
        return True

    @property
    def _support_per_replica_values(self):
        return False

    def _get_local_replica_id(self, replica_id_in_sync_group):
        return replica_id_in_sync_group