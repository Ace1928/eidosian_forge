import dataclasses
import operator
from typing import Any, List, Optional, Sequence, Tuple
from tensorflow.dtensor.python import api
from tensorflow.dtensor.python import config
from tensorflow.dtensor.python import layout as layout_lib
from tensorflow.python.data.experimental.ops import data_service_ops
from tensorflow.python.data.experimental.ops import distribute
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import iterator_ops
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_spec
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.types import data as data_types
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import tf_export
@tf_export('experimental.dtensor.DTensorDataset', v1=[])
class DTensorDataset(dataset_ops.UnaryUnchangedStructureDataset):
    """A dataset of DTensors.

  DTensorDataset encapsulates a `tf.data.Dataset` whose elements are
  automatically packed and returned as DTensors based on a given mesh and
  layouts.
  """

    def __init__(self, dataset: data_types.DatasetV2, *, mesh: layout_lib.Mesh, layouts: Any, global_batch_size: int, dataset_already_batched: bool=False, batch_dim: Optional[str]=None, prefetch: Optional[int]=None, tf_data_service_config: Optional[TFDataServiceConfig]=None):
        """Creates a DTensorDataset.

    DTensorDataset automatically handles distribution of the dataset elements to
    each client's devices. It can be used to create an iterator that returns
    DTensors of the input data on each iteration.

    DTensorDataset works best with unbatched datasets. It takes the mesh and the
    provided layouts to automatically calculate how to batch the input locally
    for each replica.

    If the provided dataset is already batched according to the per-replica
    batch size, then `dataset_already_batched` must be set and DTensorDataset
    will check that the batch size is consistent with the intended
    `global_batch_size` using the layout information. Each replica receives a
    separate slice of the global batch, thus the per-replica batch size can be
    computed as the global batch size divided by the number of model replicas.
    For a DTensor mesh, the number of replicas is equal to the size of the
    mesh's batch dimension.

    Note: `tf.experimental.dtensor.DTensorDataset` instances do *not* implement
    the full interface of `tf.data.Dataset`. It only supports two usages we will
    mention below: iteration and `element_spec`. We don't support any other APIs
    to transform or inspect the dataset.

    TODO(b/223275517): add support for input datasets that are already batched
    to the global batch size.

    Args:
      dataset: a `tf.data.Dataset` object.
      mesh: the DTensor mesh to place the dataset batches on.
      layouts: a structure of DTensor layouts to be applied to the input dataset
        values. This can be a single layout or (possibly nested) tuples or
        dictionaries of layouts, and the structure must match the structure of
        the dataset. Either all or none of the layouts should be sharded on the
        batch dimension; having only a subset of layouts batch sharded will not
        work and raises a ValueError.
      global_batch_size: the desired global batch size.
      dataset_already_batched: must be set only if the dataset is already
        batched to the per-replica batch size. The batched dataset must have
        `drop_remainder=True` set since DTensor requires static shapes for
        slicing the input tensors.
      batch_dim: the mesh dimension on which the input's batch dimension is
        sharded. Set to None if the input layouts do not shard on the batch
        dimension.
      prefetch: number of batches to prefetch using Dataset.prefetch.
      tf_data_service_config: if operating in multi-client mode, this config
        specifies the tf.data service configuration to use.

    Raises:
      ValueError: on any of the following situations,
        1. if the structures and ranks of layouts and the dataset do not match.
        2. if the shapes in the dataset's spec are not fully defined.
        3. if batch_dim is specified and all layouts are not batch-sharded.
        4. if per_replica_batch_size is specified for an already batched Dataset
           but it does not match the expected per-replica size based on the
           provided mesh.
      TypeError: if type of structures of layouts and the dataset do not match.
    """
        super().__init__(dataset, dataset_ops.to_variant(dataset))
        if tf_data_service_config is not None:
            raise NotImplementedError('Multi-client DTensorDataset is currently not supported. Check b/271162918.')
        self._mesh = mesh
        self._layouts = layouts
        self._batch_dim = batch_dim
        self._prefetch = prefetch
        self._tf_data_service_config = tf_data_service_config
        nest.assert_same_structure(dataset.element_spec, layouts)
        flattened_layouts = nest.flatten(layouts)
        flattened_elem_spec = nest.flatten(dataset.element_spec)
        if batch_dim:
            self.num_global_replicas = mesh.dim_size(batch_dim)
            self._local_replica_ids = list(dict.fromkeys([loc[batch_dim] for loc in mesh.local_device_locations()]))
            for layout in flattened_layouts:
                if batch_dim != layout.sharding_specs[0]:
                    raise ValueError('batch_dim %s was specified but at least one layout did not contain it: %s' % (batch_dim, layout))
        else:
            self.num_global_replicas = 1
            self._local_replica_ids = [0]
        _validate_input(flattened_layouts, flattened_elem_spec, dataset_already_batched=dataset_already_batched)
        expected_batch_size = global_batch_size // self.num_global_replicas
        if not dataset_already_batched:
            self._batched_dataset = dataset.batch(expected_batch_size, drop_remainder=True)
        else:
            per_replica_batch_size = flattened_elem_spec[0].shape.as_list()[0]
            if per_replica_batch_size != expected_batch_size:
                raise ValueError('per_replica_batch_size does not matched expected size based on the mesh, got %d but expected %d.' % (per_replica_batch_size, expected_batch_size))
            self._batched_dataset = dataset
        flattened_global_elem_spec = []
        batch_tensor_shape = tensor_shape.as_shape([global_batch_size])
        for elem_spec in nest.flatten(self._batched_dataset.element_spec):
            new_elem_spec = tensor_spec.TensorSpec(shape=operator.concat(batch_tensor_shape, elem_spec.shape[1:]), dtype=elem_spec.dtype, name=elem_spec.name)
            flattened_global_elem_spec.append(new_elem_spec)
        self._global_element_spec = nest.pack_sequence_as(dataset.element_spec, flattened_global_elem_spec)
        num_global_devices_per_replica = config.num_global_devices(mesh.device_type()) // self.num_global_replicas
        self._num_local_replicas = len(self._local_replica_ids)
        self._num_local_devices_per_replica = mesh.num_local_devices() // self._num_local_replicas
        self._num_clients_per_replica = num_global_devices_per_replica // self._num_local_devices_per_replica
        self._partition_offset = config.client_id() % self._num_clients_per_replica * self._num_local_devices_per_replica
        self._all_shard_counts = [_shard_counts(layout, batch_dim) for layout in flattened_layouts]
        self._index_matrices = [_index_matrix(layout, elem_spec) for layout, elem_spec in zip(flattened_layouts, flattened_elem_spec)]

    def __iter__(self):
        datasets: List[Tuple[int, data_types.DatasetV2]] = []
        local_dataset = self._batched_dataset
        if self._batch_dim is not None:
            if self._num_clients_per_replica > 1:
                local_dataset = self._repeat_batch(local_dataset, self._num_clients_per_replica)
                sharding_policy = data_service_ops.ShardingPolicy.DATA
            else:
                sharding_policy = data_service_ops.ShardingPolicy.FILE
        else:
            sharding_policy = data_service_ops.ShardingPolicy.OFF
        if self._tf_data_service_config is not None:
            local_dataset = local_dataset.apply(data_service_ops.distribute(processing_mode=sharding_policy, service=self._tf_data_service_config.dispatcher_address, job_name=f'{self._tf_data_service_config.job_name}_{config.client_id()}', target_workers='LOCAL'))
        for local_replica_idx, replica_id in enumerate(self._local_replica_ids):
            dataset = distribute._AutoShardDataset(local_dataset, num_workers=self._num_local_replicas, index=local_replica_idx, num_replicas=self.num_global_replicas)
            dataset = self._repeat_batch(dataset, self._num_local_devices_per_replica)
            dataset = self._partition(dataset)
            if self._prefetch is not None:
                dataset = dataset.prefetch(self._prefetch * self._num_local_devices_per_replica)
            datasets.append((replica_id, dataset))
        d_iterator_resource = _pack_iterator_resource_dtensor(datasets=datasets, layouts=self._layouts, mesh=self._mesh, num_local_devices_per_replica=self._num_local_devices_per_replica)
        return _DTensorIterator(dtensor_components=(d_iterator_resource,), global_element_spec=self._global_element_spec, layouts=self._layouts)

    def _repeat_batch(self, dataset, repeats):
        if repeats == 1:
            return dataset

        def repeat(*x):
            return dataset_ops.DatasetV2.from_tensors(x).repeat(repeats)
        return dataset.flat_map(repeat)

    def _partition(self, dataset):
        """Slices each dataset element on any sharded non-batch dimension."""
        if self._num_local_devices_per_replica == 1 and self._partition_offset == 0:
            return dataset

        def slice_batch(index, batch):
            flattened_batch = nest.flatten(batch)
            flattened_output = []
            norm_index = math_ops.cast(index % self._num_local_devices_per_replica, dtype=dtypes.int32)
            norm_index += self._partition_offset
            coords = self._mesh.coords(norm_index)
            coords = array_ops.reshape(coords, (1, -1))
            for element, shard_counts, idx_matrix in zip(flattened_batch, self._all_shard_counts, self._index_matrices):
                indexes = math_ops.matmul(coords, idx_matrix)
                start = array_ops.reshape(indexes, (-1,))
                size = array_ops.shape_v2(element, out_type=dtypes.int32) // shard_counts
                flattened_output.append(array_ops.slice(element, begin=start, size=size))
            return nest.pack_sequence_as(batch, flattened_output)
        enumerated_dataset = dataset.enumerate()
        partitioned_dataset = enumerated_dataset.map(slice_batch)
        return partitioned_dataset

    @property
    def element_spec(self):
        return self._global_element_spec