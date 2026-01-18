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
@tf_export('experimental.dtensor.Mesh', v1=[])
class Mesh(_pywrap_dtensor_device.Mesh):
    """Represents a Mesh configuration over a certain list of Mesh Dimensions.

  A mesh consists of named dimensions with sizes, which describe how a set of
  devices are arranged. Defining tensor layouts in terms of mesh dimensions
  allows us to efficiently determine the communication required when computing
  an operation with tensors of different layouts.

  A mesh provides information not only about the placement of the tensors but
  also the topology of the underlying devices. For example, we can group 8 TPUs
  as a 1-D array for data parallelism or a `2x4` grid for (2-way) data
  parallelism and (4-way) model parallelism.

  Refer to [DTensor Concepts](https://www.tensorflow.org/guide/dtensor_overview)
  for in depth discussion and examples.

  Note: the utilities `dtensor.create_mesh` and
  `dtensor.create_distributed_mesh` provide a simpler API to create meshes for
  single- or multi-client use cases.
  """

    def __init__(self, dim_names: List[str], global_device_ids: np.ndarray, local_device_ids: List[int], local_devices: List[Union[tf_device.DeviceSpec, str]], mesh_name: str='', global_devices: Optional[List[Union[tf_device.DeviceSpec, str]]]=None, use_xla_spmd: bool=USE_XLA_SPMD):
        """Builds a Mesh.

    The `dim_names` and `global_device_ids` arguments describe the dimension
    names and shape for the mesh.

    For example,

    ```python
      dim_names = ('x', 'y'),
      global_device_ids = [[0, 1],
                           [2, 3],
                           [4, 5]]
    ```

    defines a 2D mesh of shape 3x2. A reduction over the 'x' dimension will
    reduce across columns (0, 2, 4) and (1, 3, 5), and a reduction over the 'y'
    dimension reduces across rows.

    Note: the utilities `dtensor.create_mesh` and
    `dtensor.create_distributed_mesh` provide a simpler API to create meshes for
    single- or multi-client use cases.

    Args:
      dim_names: A list of strings indicating dimension names.
      global_device_ids: An ndarray of global device IDs is used to compose
        DeviceSpecs describing the mesh. The shape of this array determines the
        size of each mesh dimension. Values in this array should increment
        sequentially from 0. This argument is the same for every DTensor client.
      local_device_ids: A list of local device IDs equal to a subset of values
        in global_device_ids. They indicate the position of local devices in the
        global mesh. Different DTensor clients must contain distinct
        local_device_ids contents. All local_device_ids from all DTensor clients
        must cover every element in global_device_ids.
      local_devices: The list of devices hosted locally. The elements correspond
        1:1 to those of local_device_ids.
      mesh_name: The name of the mesh. Currently, this is rarely used, and is
        mostly used to indicate whether it is a CPU, GPU, or TPU-based mesh.
      global_devices (optional): The list of global devices. Set when multiple
        device meshes are in use.
      use_xla_spmd (optional): Boolean when True, will use XLA SPMD instead of
        DTensor SPMD.
    """
        if not isinstance(global_device_ids, np.ndarray):
            raise ValueError('Variable global_device_ids must be an ndarray.')
        if global_device_ids.size == 0:
            raise ValueError('Variable global_device_ids must be non-empty.')
        flat_global_device_ids = global_device_ids.flatten()
        distance = flat_global_device_ids[0]
        if any((gid - i != distance for i, gid in enumerate(flat_global_device_ids))):
            raise ValueError('global_device_ids must sequentially increase: %s' % global_device_ids)
        if len(dim_names) != global_device_ids.ndim:
            raise ValueError('Number of mesh dimensions does not match number of dimension names.')
        if not isinstance(local_device_ids, list):
            raise ValueError('Variable local_device_ids must be a list of integers.')
        if not isinstance(local_devices, list):
            raise ValueError('Variable local_devices must be a list of DeviceSpecs.')
        if global_devices and (not isinstance(global_devices, list)):
            raise ValueError('Variable global_devices must be a list of DeviceSpecs.')
        if not local_devices and (not global_devices):
            raise ValueError('Empty list of devices not allowed.')
        global_device_ids_flatten = global_device_ids.flatten()
        global_device_ids_shape = global_device_ids.shape

        def to_str(d) -> str:
            if isinstance(d, tf_device.DeviceSpec):
                return d.to_string()
            return d

        def to_spec(d) -> tf_device.DeviceSpec:
            if not isinstance(d, tf_device.DeviceSpec):
                return tf_device.DeviceSpec.from_string(d)
            return d
        local_devices_str = [to_str(d) for d in local_devices]
        local_devices_spec = [to_spec(d) for d in local_devices]
        if not global_devices:
            global_devices = []
        global_devices_str = [to_str(d) for d in global_devices]
        global_devices_spec = [to_spec(d) for d in global_devices]
        local_devices_set = set(local_devices_spec)
        local_device_only_contains_host_cpu = len(local_devices_set) == 1 and list(local_devices_set)[0].device_type == 'CPU'
        if not local_device_only_contains_host_cpu and len(local_devices) != len(local_devices_set):
            raise ValueError('Duplicate devices found in mesh specification %s.' % [d for d in local_devices if local_devices.count(d) > 1])
        if len(local_device_ids) != len(local_devices):
            raise ValueError('Variable local_device_ids does not have same size as local_devices.')
        if len(local_device_ids) > len(np.ravel(global_device_ids)):
            raise ValueError('Cannot have more local than gobal device IDs.')
        device_types = set([device.device_type for device in local_devices_spec])
        if not device_types:
            device_types = set([device.device_type for device in global_devices_spec])
        if None in device_types:
            raise ValueError('device_type is required')
        if len(device_types) > 1:
            raise ValueError('Devices containing multiple device_types : %s' % device_types)
        device_type = device_types.pop()
        if use_xla_spmd and device_type != 'TPU':
            raise ValueError('XLA SPMD is not currently not supported for %s mesh.' % device_type)
        super().__init__(mesh_name, dim_names, global_device_ids_shape, global_device_ids_flatten, global_devices_str, local_device_ids, local_devices_str, use_xla_spmd)

    @classmethod
    def _new_object(cls, *args, **kwargs):
        self = _pywrap_dtensor_device.Mesh.__new__(cls)
        super().__init__(self, *args, **kwargs)
        return self

    def global_device_ids(self) -> np.ndarray:
        """Returns a global device list as an array."""
        return np.array(super().global_device_ids(), dtype=np.int64).reshape(self.shape())

    def __getitem__(self, dim_name: str) -> MeshDimension:
        return MeshDimension(name=dim_name, size=self.dim_size(dim_name))

    def __hash__(self):
        return hash(self.as_proto().SerializeToString(deterministic=True))

    def __repr__(self) -> str:
        return f'Mesh.from_string({self.to_string()})'

    def __reduce__(self):
        return (Mesh.from_string, (self.to_string(),))

    def coords(self, device_idx: int) -> tensor.Tensor:
        """Converts the device index into a tensor of mesh coordinates."""
        strides = ops.convert_to_tensor(self.strides)
        shape = ops.convert_to_tensor(self.shape())
        return device_idx // strides % shape

    @classmethod
    def from_proto(cls, proto: layout_pb2.MeshProto) -> 'Mesh':
        """Construct a mesh instance from input `proto`."""
        return cls._new_object(mesh_proto=proto)

    @classmethod
    def from_string(cls, mesh_str: str) -> 'Mesh':
        return cls._new_object(mesh_str=mesh_str)

    @classmethod
    def from_device(cls, device: str) -> 'Mesh':
        """Constructs a single device mesh from a device string."""
        return cls._new_object(single_device=device)

    @classmethod
    def _from_mesh(cls, mesh: _pywrap_dtensor_device.Mesh):
        """Creates a copy from an existing pywrap mesh object."""
        return cls._new_object(mesh=mesh)

    @functools.cached_property
    def _host_mesh(self) -> 'Mesh':
        return Mesh._from_mesh(super().host_mesh())

    def host_mesh(self) -> 'Mesh':
        """Returns a host mesh."""
        return self._host_mesh

    def local_device_locations(self) -> List[Dict[str, int]]:
        """Returns a list of local device locations.

    A device location is a dictionary from dimension names to indices on those
    dimensions.
    """
        mapping = self.unravel_index()
        return [mapping[device_id] for device_id in self.local_device_ids()]

    @property
    def strides(self) -> List[int]:
        """Returns the strides tensor array for this mesh.

    If the mesh shape is `[a, b, c, d]`, then the strides array can be computed
    as `[b*c*d, c*d, d, 1]`. This array can be useful in computing local device
    offsets given a device ID. Using the same example, the device coordinates of
    the mesh can be computed as:

    ```
    [(device_id / (b*c*d)) % a,
     (device_id / (c*d))   % b,
     (device_id / (d))     % c,
     (device_id)           % d]
    ```

    This is the same as `(device_id // mesh.strides) % mesh.shape`.

    Returns:
      The mesh strides as an integer tensor.
    """
        return _compute_mesh_strides(self.shape())

    def unravel_index(self):
        """Returns a dictionary from device ID to {dim_name: dim_index}.

    For example, for a 3x2 mesh, return this:

    ```
      { 0: {'x': 0, 'y', 0},
        1: {'x': 0, 'y', 1},
        2: {'x': 1, 'y', 0},
        3: {'x': 1, 'y', 1},
        4: {'x': 2, 'y', 0},
        5: {'x': 2, 'y', 1} }
    ```
    """
        idx_ranges = [range(self.dim_size(dim_name)) for dim_name in self.dim_names]
        mesh_pos = itertools.product(*idx_ranges)
        mapping = {}
        for device_id, device_pos in enumerate(mesh_pos):
            device_loc = {}
            for dim_name, dim_index in zip(self.dim_names, device_pos):
                device_loc[dim_name] = dim_index
            mapping[device_id] = device_loc
        return mapping