from typing import Any, List
import torch
from torch.distributed._shard.metadata import ShardMetadata
from torch.distributed._shard.sharded_tensor import ShardedTensor
from torch.distributed._shard.sharded_tensor.metadata import TensorProperties
from torch.distributed._tensor import DTensor
from torch.distributed._tensor._utils import compute_local_shape_and_global_offset
from .metadata import (
from .planner import (
from .resharding import (
def _create_chunk_from_dtensor(tensor: DTensor) -> ChunkStorageMetadata:
    sizes, offsets = compute_local_shape_and_global_offset(tensor.shape, tensor.device_mesh, tensor.placements)
    sizes, offsets = (torch.Size(sizes), torch.Size(offsets))
    return ChunkStorageMetadata(offsets=offsets, sizes=sizes)