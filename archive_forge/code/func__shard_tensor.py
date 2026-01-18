from dataclasses import dataclass
from typing import Any, cast, List, NamedTuple, Optional, Tuple
import torch
import torch.distributed._functional_collectives as funcol
import torch.distributed.distributed_c10d as c10d
from torch.distributed._tensor._collective_utils import mesh_broadcast, mesh_scatter
from torch.distributed.device_mesh import DeviceMesh
def _shard_tensor(self, tensor: torch.Tensor, mesh: DeviceMesh, mesh_dim: int) -> torch.Tensor:
    """
        shard and scatter a tensor on a mesh dimension (use coordinate
        0 on the mesh dimension as source of truth)
        """
    my_coordinate = mesh.get_coordinate()
    num_chunks = mesh.size(mesh_dim=mesh_dim)
    if my_coordinate is None:
        return tensor.new_empty(0, requires_grad=tensor.requires_grad)
    scatter_list, pad_sizes = self._split_tensor(tensor, num_chunks, with_padding=True, contiguous=True)
    output = torch.empty_like(scatter_list[my_coordinate[mesh_dim]])
    mesh_scatter(output, scatter_list, mesh, mesh_dim=mesh_dim)
    pad_size = pad_sizes[my_coordinate[mesh_dim]]
    if pad_size > 0:
        output = self._unpad_tensor(output, pad_size)
    return output