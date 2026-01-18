from dataclasses import dataclass
from typing import Any, cast, List, NamedTuple, Optional, Tuple
import torch
import torch.distributed._functional_collectives as funcol
import torch.distributed.distributed_c10d as c10d
from torch.distributed._tensor._collective_utils import mesh_broadcast, mesh_scatter
from torch.distributed.device_mesh import DeviceMesh
@dataclass(frozen=True)
class _Partial(Placement):
    reduce_op: c10d.ReduceOp.RedOpType = c10d.ReduceOp.SUM

    def _to_replicate(self, tensor: torch.Tensor, mesh: DeviceMesh, mesh_dim: int) -> torch.Tensor:
        return funcol.all_reduce(tensor, reduceOp=self.reduce_op.name, group=(mesh, mesh_dim))

    def _to_shard(self, tensor: torch.Tensor, mesh: DeviceMesh, mesh_dim: int, shard_spec: Placement) -> torch.Tensor:
        shard_spec = cast(Shard, shard_spec)
        return shard_spec._reduce_shard_tensor(tensor, mesh, self.reduce_op, mesh_dim)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, _Partial):
            return False
        return self.reduce_op == other.reduce_op

    def __hash__(self) -> int:
        return 1 + hash(self.reduce_op)

    def __repr__(self) -> str:
        """
        machine readable representation of the Partial placement
        """
        return f'_Partial(reduce_op={self.reduce_op})'

    def __str__(self) -> str:
        """
        human readable representation of the Partial placement
        """
        return 'P'