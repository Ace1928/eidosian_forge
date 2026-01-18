from dataclasses import dataclass
from typing import Any, cast, List, NamedTuple, Optional, Tuple
import torch
import torch.distributed._functional_collectives as funcol
import torch.distributed.distributed_c10d as c10d
from torch.distributed._tensor._collective_utils import mesh_broadcast, mesh_scatter
from torch.distributed.device_mesh import DeviceMesh
def _hash_impl(self) -> int:
    if self.tensor_meta is not None:
        return hash((self.mesh, self.placements, self.tensor_meta.shape, self.tensor_meta.stride, self.tensor_meta.dtype))
    return hash((self.mesh, self.placements))