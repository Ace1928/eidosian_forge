import copy
import itertools
import math
from typing import Optional
import torch
import torch.distributed as dist
from torch.distributed import distributed_c10d
from torch.distributed._shard.sharded_tensor import (
from torch.distributed._shard.sharding_spec import ShardMetadata
from torch.distributed._tensor import DeviceMesh, DTensor, Replicate, Shard as DShard
def _create_chunk_dtensor(tensor: torch.Tensor, rank: int, device_mesh: DeviceMesh) -> DTensor:
    """
    Shard a tensor to chunks along the first dimension. The local rank will gets its
    corresponding chunk as the local tensor to create a DTensor.
    """
    tensor = tensor.clone().detach()
    replicate_placements = [Replicate() for _ in range(device_mesh.ndim)]
    shard_placements = [Replicate() for _ in range(device_mesh.ndim)]
    shard_placements[-1] = DShard(0)
    return DTensor.from_local(tensor, device_mesh, replicate_placements, run_check=False).redistribute(placements=shard_placements)