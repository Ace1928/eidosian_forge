from dataclasses import dataclass
from typing import Any, cast, List, NamedTuple, Optional, Tuple
import torch
import torch.distributed._functional_collectives as funcol
import torch.distributed.distributed_c10d as c10d
from torch.distributed._tensor._collective_utils import mesh_broadcast, mesh_scatter
from torch.distributed.device_mesh import DeviceMesh
def _to_shard(self, tensor: torch.Tensor, mesh: DeviceMesh, mesh_dim: int, shard_spec: Placement) -> torch.Tensor:
    shard_spec = cast(Shard, shard_spec)
    return shard_spec._reduce_shard_tensor(tensor, mesh, self.reduce_op, mesh_dim)