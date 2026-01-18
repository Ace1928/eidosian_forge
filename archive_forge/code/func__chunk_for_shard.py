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
def _chunk_for_shard(shard_md: ShardMetadata) -> ChunkStorageMetadata:
    return ChunkStorageMetadata(offsets=torch.Size(shard_md.shard_offsets), sizes=torch.Size(shard_md.shard_sizes))