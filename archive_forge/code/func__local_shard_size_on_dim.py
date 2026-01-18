from dataclasses import dataclass
from typing import Any, cast, List, NamedTuple, Optional, Tuple
import torch
import torch.distributed._functional_collectives as funcol
import torch.distributed.distributed_c10d as c10d
from torch.distributed._tensor._collective_utils import mesh_broadcast, mesh_scatter
from torch.distributed.device_mesh import DeviceMesh
def _local_shard_size_on_dim(self, size_on_dim: int, num_chunks: int, rank: int, return_offset: bool=False) -> Tuple[int, int]:
    """
        returns the local shard size and offset on a given tensor dim
        """
    assert size_on_dim >= num_chunks, f'Size to be sharded on dim {self.dim} must be at least as large as the number of devices in that dimension {num_chunks}'
    full_chunk_size = (size_on_dim + num_chunks - 1) // num_chunks
    chunk_sizes = [max(min(size_on_dim, full_chunk_size * (idx + 1)) - full_chunk_size * idx, 0) for idx in range(num_chunks)]
    local_shard_size = chunk_sizes[rank]
    local_offset_on_dim = -1
    if return_offset:
        local_offset_on_dim = sum(chunk_sizes[:rank])
    return (local_shard_size, local_offset_on_dim)