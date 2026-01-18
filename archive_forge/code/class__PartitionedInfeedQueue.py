import itertools
import numpy as np
from tensorflow.python.compiler.xla.experimental import xla_sharding
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.tpu import tpu_name_util
from tensorflow.python.tpu import tpu_sharding
from tensorflow.python.tpu.ops import tpu_ops
from tensorflow.python.util import nest
class _PartitionedInfeedQueue(InfeedQueue):
    """A helper object to build a device infeed queue with input partition.

  Args:
    number_of_tuple_elements: the number of Tensors fed atomically through the
      queue, must be present unless it can be inferred from other arguments.
    device_assignment: A TPU `DeviceAssignment` which is used to place all the
      partitions to different TPU infeed queues.
    host_id: The id of the host machine.
    input_partition_dims: A nested list/tuple of integers. Each inner
      list/tuple describes how to partition the corresponding input tensor.
    tuple_types: If not None, a list of types of the elements of the queue.
    tuple_shapes: If not None, a list of shapes of the elements of the queue.
    name: The name of the queue.
  """

    def __init__(self, number_of_tuple_elements, device_assignment, host_id, input_partition_dims=None, tuple_types=None, tuple_shapes=None, name=None):
        super(_PartitionedInfeedQueue, self).__init__(number_of_tuple_elements=number_of_tuple_elements, tuple_types=tuple_types, tuple_shapes=None, shard_dimensions=None, name='PartitionedInfeedQueue' if name is None else name)
        self._input_partition_dims = input_partition_dims
        self._host_id = host_id
        self._device_assignment = device_assignment

    def generate_dequeue_op(self, tpu_device=0):
        """Generate TPU dequeue ops.

    Args:
      tpu_device: The TPU device ordinal where the infeed instruction should be
        placed.

    Returns:
      A list of Outputs corresponding to a partition of infeed dequeued
      into XLA, suitable for use within a replicated block.

    Raises:
      ValueError: if the types or shapes of the tuple elements have not been
      set; or if a dequeue op has already been generated.
    """
        self.freeze()
        if self._generated_dequeue_op and (not ops.inside_function()):
            raise ValueError("Can't generate two dequeue Ops from the same queue")
        self._generated_dequeue_op = True
        full_name = '%s/dequeue' % self._name
        sharded_shapes = [policy.get_sharded_shape(shape) for shape, policy in zip(self._tuple_shapes, self._sharding_policies)]
        with ops.device(tpu_name_util.core(tpu_device)):
            values = tpu_ops.infeed_dequeue_tuple(dtypes=self._tuple_types, shapes=sharded_shapes, name=full_name)
        return tag_sharding_attribute_for_dequeued_tensors(values, self._input_partition_dims)

    def generate_enqueue_ops(self, sharded_inputs):
        """Generates the host-side Ops to enqueue the partitioned inputs.

    sharded_inputs is a list, one for each replica, of lists of
    Tensors. sharded_inputs[i] is the tuple of Tensors to use to feed
    replica i.
    sharded_inputs[i][j] is partitioned by self._input_partition_dims[j].

    For example, if sharded_inputs[i][j] is a 2-D Tensor:
    [[A, B, C, D],
     [E ,F, G, H]]
    self._input_partition_dims[j] is [2, 4].

    sharded_inputs[i][j] will be partitioned and flattened into:
    [A, B, C, D, E, F, G, H] and fed into the logical core ids:
    [0, 1, 2, 3, 4, 5, 6, 7] respectively.

    Args:
      sharded_inputs: a list of lists of Tensors. The length of the
        outer list determines the number of shards. Each inner list indicates
        the types and shapes of the tuples in the corresponding shard.

    Returns:
      A list of host-side Ops, one for each shard, that when executed together
      will enqueue a full-size element of infeed.

    Raises:
      ValueError: if the queue configuration has previously been frozen and the
        shapes of the elements of sharded_inputs are not compatible with the
        frozen configuration; or if the shapes of the elements of sharded_inputs
        don't form a consistent unsharded tuple; or if the elements of a tuple
        have different device constraints; or if the partition dims are invalid.
      TypeError: if the queue configuration has previously been frozen and the
        types of the elements of sharded_inputs are not compatible with the
        frozen configuration; or if the types of the elements of sharded_inputs
        don't form a consistent unsharded tuple.
    """
        self.set_configuration_from_sharded_input_tensors(sharded_inputs)
        number_of_replicas = len(sharded_inputs)
        number_of_tuple_elements = len(sharded_inputs[0])
        assert len(self._input_partition_dims) == number_of_tuple_elements
        enqueue_ops = []
        for replica_index in range(number_of_replicas):
            flattened_inputs = sharded_inputs[replica_index]
            inputs_part_dims_flat = nest.flatten_up_to(flattened_inputs, self._input_partition_dims)
            inputs_parted_iters = [iter(self._check_dims_and_partition_or_replicate_on_host(x, dims)) for x, dims in zip(sharded_inputs[replica_index], inputs_part_dims_flat)]
            replica_id = self._device_assignment.lookup_replicas(task_id=self._host_id, logical_core=0)[replica_index]
            for logical_core in range(self._device_assignment.num_cores_per_replica):
                device = self._device_assignment.host_device(replica=replica_id, logical_core=logical_core)
                with ops.device(device):
                    ordinal = self._device_assignment.tpu_ordinal(replica=replica_id, logical_core=logical_core)
                    infeed_inputs = []
                    for it in inputs_parted_iters:
                        input_for_device = next(it, None)
                        if input_for_device is not None:
                            infeed_inputs.append(input_for_device)
                    if infeed_inputs:
                        enqueue_ops.append(tpu_ops.infeed_enqueue_tuple(inputs=infeed_inputs, shapes=[x.shape for x in infeed_inputs], name='enqueue/replica_{0}/input_{1}'.format(replica_index, logical_core), device_ordinal=ordinal))
        return enqueue_ops

    def _check_input_partition_dims(self, tensor, dims):
        """Checks that input partition dims are valid for the `Tensor`.

    Args:
      tensor: Input tensor for partitioning.
      dims: A list of integer describes how to partition the input tensor.

    Raises:
      ValueError: If the tensor can't be partitioned by dims or the
        num_cores_per_replica doesn't match the number of
        partitions(dims.prod()).
    """
        if dims is None:
            return
        dims = np.array(dims)
        if (dims < 1).any():
            raise ValueError('All input partition dims must be >= 1.')
        if dims.prod() == 1:
            return
        if dims.prod() != self._device_assignment.num_cores_per_replica:
            raise ValueError('The product of each input partition dim should equal to num_cores_per_replica. (dim = {}, num_cores_per_replica = {})'.format(dims, self._device_assignment.num_cores_per_replica))
        if dims.shape[0] != tensor.shape.ndims:
            raise ValueError('Input partition dims must have the same number of dimensions as the `Tensor` to be partitioned. (tensor shape = {}, input partition dims = {}).'.format(tensor.shape.as_list(), dims))
        tensor.shape.assert_is_fully_defined()

    def _check_dims_and_partition_or_replicate_on_host(self, tensor, dims):
        """Checks dims and partitions or replicates the input tensor.

      The ops inside this function are placed on the host side.

    Args:
      tensor: The input tensor which will be partitioned or replicated.
      dims: A list of integer describes how to partition the input tensor.

    Returns:
      An iterator of `Tensor`s or a list of partitioned tensors.
    """
        self._check_input_partition_dims(tensor, dims)
        return partition_or_replicate_on_host(tensor, dims)