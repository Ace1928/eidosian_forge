import collections
import functools
import itertools
from typing import List, Dict, Optional, Union
import numpy as np
from tensorflow.dtensor.proto import layout_pb2
from tensorflow.python import _pywrap_dtensor_device
from tensorflow.python.framework import device as tf_device
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.util.tf_export import tf_export
@tf_export('experimental.dtensor.Layout', v1=[])
class Layout(_pywrap_dtensor_device.Layout):
    """Represents the layout information of a DTensor.

  A layout describes how a distributed tensor is partitioned across a mesh (and
  thus across devices). For each axis of the tensor, the corresponding
  sharding spec indicates which dimension of the mesh it is sharded over. A
  special sharding spec `UNSHARDED` indicates that axis is replicated on
  all the devices of that mesh.

  Refer to [DTensor Concepts](https://www.tensorflow.org/guide/dtensor_overview)
  for in depth discussion and examples.

  For example, let's consider a 1-D mesh:

  ```
  Mesh(["TPU:0", "TPU:1", "TPU:2", "TPU:3", "TPU:4", "TPU:5"], [("x", 6)])
  ```

  This mesh arranges 6 TPU devices into a 1-D array. `Layout([UNSHARDED], mesh)`
  is a layout for rank-1 tensor which is replicated on the 6 devices.

  For another example, let's consider a 2-D mesh:

  ```
  Mesh(["TPU:0", "TPU:1", "TPU:2", "TPU:3", "TPU:4", "TPU:5"],
       [("x", 3), ("y", 2)])
  ```

  This mesh arranges 6 TPU devices into a `3x2` 2-D array.
  `Layout(["x", UNSHARDED], mesh)` is a layout for rank-2 tensor whose first
  axis is sharded on mesh dimension "x" and the second axis is replicated. If we
  place `np.arange(6).reshape((3, 2))` using this layout, the individual
  components tensors would look like:

  ```
  Device  |  Component
   TPU:0     [[0, 1]]
   TPU:1     [[0, 1]]
   TPU:2     [[2, 3]]
   TPU:3     [[2, 3]]
   TPU:4     [[4, 5]]
   TPU:5     [[4, 5]]
  ```
  """

    def __init__(self, sharding_specs: List[str], mesh: Mesh):
        """Builds a Layout from a list of dimension names and a Mesh.

    Args:
      sharding_specs: List of sharding specifications, each corresponding to a
        tensor axis. Each specification (dim_sharding) can either be a mesh
        dimension or the special value UNSHARDED.
      mesh: A mesh configuration for the Tensor.

    Returns:
      A valid Layout built with given layout & mesh.
    """
        if not isinstance(mesh, Mesh):
            raise ValueError('mesh is not a valid Mesh object.')
        for _, dim_sharding in enumerate(sharding_specs):
            if dim_sharding == UNSHARDED or dim_sharding == MATCH:
                continue
            if sharding_specs.count(dim_sharding) > 1:
                raise ValueError(('Mesh dimension {mesh_dim} was repeated in sharding ' + 'specification {sharding_specs}. Mesh dimensions must be unique ' + 'in a layout.').format(mesh_dim=dim_sharding, sharding_specs=sharding_specs))
            if dim_sharding not in mesh:
                raise ValueError(('{dim_sharding}: A dimension sharding must either be a ' + 'valid mesh dimension or UNSHARDED.').format(dim_sharding=dim_sharding))
        super().__init__(type=LayoutType.STATIC, sharding_specs=sharding_specs, mesh=mesh)

    @classmethod
    def _new_object(cls, *args, **kwargs):
        self = _pywrap_dtensor_device.Layout.__new__(cls)
        super().__init__(self, *args, **kwargs)
        return self

    def __repr__(self) -> str:
        return f'Layout.from_string({self.to_string()})'

    def __hash__(self):
        return hash(self.as_proto().SerializeToString(deterministic=True))

    def __reduce__(self):
        return (Layout.from_string, (self.to_string(),))

    @property
    def mesh(self):
        return Mesh._from_mesh(mesh=super().mesh)

    @property
    def shape(self):
        return self.mesh.shape()

    @classmethod
    def batch_sharded(cls, mesh: Mesh, batch_dim: str, rank: int, axis: int=0) -> 'Layout':
        """Returns a layout sharded on batch dimension."""
        return cls._new_object(mesh=mesh, rank=rank, batch_dim=batch_dim, axis=axis)

    def delete(self, dims: List[int]) -> 'Layout':
        """Returns the layout with the give dimensions deleted."""
        if not isinstance(dims, list):
            dims = [dims]
        new_specs = [spec for i, spec in enumerate(self.sharding_specs) if i not in dims]
        return Layout(new_specs, self.mesh)

    @classmethod
    def from_proto(cls, layout_proto: layout_pb2.LayoutProto) -> 'Layout':
        """Creates an instance from a LayoutProto."""
        return cls._new_object(layout_proto=layout_proto)

    @classmethod
    def from_string(cls, layout_str: str) -> 'Layout':
        """Creates an instance from a human-readable string."""
        return cls._new_object(layout_str=layout_str)

    def to_parted(self) -> 'Layout':
        """Returns a "parted" layout from a static layout.

    A parted layout contains axes that are treated as independent by most of
    SPMD expanders.

    FIXME(b/285905569): The exact semantics is still being investigated.
    """
        return Layout._new_object(layout=super().to_parted())

    @classmethod
    def inner_sharded(cls, mesh: Mesh, inner_dim: str, rank: int) -> 'Layout':
        """Returns a layout sharded on inner dimension."""
        return cls.batch_sharded(mesh, inner_dim, rank, axis=rank - 1)

    @classmethod
    def from_single_device_mesh(cls, mesh: Mesh) -> 'Layout':
        """Constructs a single device layout from a single device mesh."""
        return cls._new_object(mesh=mesh)

    @classmethod
    def from_device(cls, device: str) -> 'Layout':
        """Constructs a single device layout from a single device mesh."""
        return cls.from_single_device_mesh(Mesh.from_device(device))

    def offset_to_shard(self):
        """Mapping from offset in a flattened list to shard index."""
        unravel_index = self.mesh.unravel_index()
        locations = [None] * self.mesh.size
        for offset, mesh_loc in unravel_index.items():
            loc = []
            for dim_sharding in self.sharding_specs:
                if dim_sharding == UNSHARDED:
                    loc.append(0)
                else:
                    loc.append(mesh_loc[dim_sharding])
            locations[offset] = tuple(loc)
        return locations

    def offset_tuple_to_global_index(self, offset_tuple):
        """Mapping from offset to index in global tensor."""
        index = 0
        for i, o in enumerate(offset_tuple):
            m = 1
            for x in range(i + 1, self.rank):
                m = m * self.num_shards(x)
            index = index + m * o
        return index

    @classmethod
    def replicated(cls, mesh: Mesh, rank: int) -> 'Layout':
        """Returns a replicated layout of rank `rank`."""
        return cls._new_object(mesh=mesh, rank=rank)