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
def _pack_iterator_resource_dtensor(datasets: List[Tuple[int, data_types.DatasetV2]], layouts: Any, mesh: layout_lib.Mesh, num_local_devices_per_replica: int):
    """Creates a DTensor iterator resource for the per-replica datasets.

  Given a list of replica ID to tf.data.Dataset mappings, this function creates
  iterators for each device and then packs the underlying iterator resource
  tensors into a single DTensor. This resource tensor is used by the
  IteratorGetNext op to retrieve the next element in the dataset.

  Args:
    datasets: a list of tuples of each unique local replica ID to the dataset
      object whose elements will be placed on the devices corresponding to that
      replica.
    layouts: a structure of DTensor layouts to be applied to the elements
      returned by the underlying iterators. This can be a single layout or
      (possibly nested) tuples or dictionaries of layouts, and the structure
      must match the structure of the iterator elements.
    mesh: the DTensor mesh to place the iterator batches on.
    num_local_devices_per_replica: the number of devices in each data-parallel
      replica.

  Returns:
    A DTensor of the underlying iterator resource tensors.
  """
    host_mesh_devices = mesh.host_mesh().local_devices()
    device_idx = 0
    iterators = []
    for _, dataset in datasets:
        for idx in range(num_local_devices_per_replica):
            with ops.device_v2(host_mesh_devices[device_idx]):
                device_dataset = dataset.shard(num_shards=num_local_devices_per_replica, index=idx)
                iterators.append(iter(device_dataset))
            device_idx += 1
    if device_idx != len(host_mesh_devices):
        raise ValueError(f'The `datasets` argument does not have the correct number of underlying datasets, found {device_idx} but expected {len(host_mesh_devices)}.')
    host_layouts = nest.map_structure(lambda l: layout_lib.Layout(l.sharding_specs, mesh.host_mesh()), layouts)
    iterator_resources = [it._iterator_resource for it in iterators]
    d_iterator_resource = api.pack(iterator_resources, layout_lib.Layout.replicated(mesh=mesh.host_mesh(), rank=0))
    api._dtensor_device().set_iterator_element_layouts(d_iterator_resource, nest.flatten(host_layouts))
    return d_iterator_resource