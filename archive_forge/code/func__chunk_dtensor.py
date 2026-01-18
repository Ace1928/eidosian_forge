import copy
from typing import Any, cast, List, Optional, Tuple
import torch
import torch.distributed as dist
import torch.distributed._shard.sharding_spec as shard_spec
import torch.distributed.distributed_c10d as c10d
from torch.distributed._shard.sharded_tensor import (
from torch.distributed._shard.sharding_spec import ShardMetadata
from torch.distributed._shard.sharding_spec.chunk_sharding_spec import ChunkShardingSpec
from torch.distributed._tensor import DeviceMesh, DTensor, Replicate, Shard as DShard
from torch.distributed.device_mesh import _mesh_resources
from torch.distributed.fsdp._common_utils import _set_fsdp_flattened
from torch.distributed.fsdp._fsdp_extensions import FSDPExtensions
from torch.distributed.fsdp._shard_utils import _create_chunk_sharded_tensor
from torch.distributed.remote_device import _remote_device
from torch.distributed.tensor.parallel._data_parallel_utils import (
def _chunk_dtensor(tensor: torch.Tensor, rank: int, device_mesh: DeviceMesh) -> DTensor:
    """
    Shard a tensor to chunks along the first dimension.

    The local rank will gets its corresponding chunk as the local tensor to create a DTensor.
    """
    parent_mesh = _mesh_resources.get_parent_mesh(device_mesh)
    if parent_mesh is None:
        raise RuntimeError('No parent device_mesh is found for FSDP device_mesh.')
    if parent_mesh.ndim < 2:
        raise RuntimeError(f'Found parent device_mesh of ndim={parent_mesh.ndim},', 'but meshes must be at least 2D.')
    tensor = tensor.clone().detach()
    if isinstance(tensor, torch.Tensor) and (not isinstance(tensor, DTensor)):
        replicate_placements = [Replicate() for _ in range(parent_mesh.ndim)]
        shard_placements = [Replicate() for _ in range(parent_mesh.ndim)]
        shard_placements[0] = DShard(0)
        return DTensor.from_local(tensor, parent_mesh, replicate_placements).redistribute(device_mesh=parent_mesh, placements=shard_placements)
    else:
        tp_placements = tensor.placements
        tp_placement = tp_placements[0]
        tensor = tensor.to_local()
        replicate_placements = [Replicate() for _ in range(parent_mesh.ndim)]
        replicate_placements[-1] = tp_placement
        shard_placements = [Replicate() for i in range(parent_mesh.ndim)]
        shard_placements[-2] = DShard(0)
        shard_placements[-1] = tp_placement
        return DTensor.from_local(tensor, parent_mesh, replicate_placements).redistribute(device_mesh=parent_mesh, placements=shard_placements)