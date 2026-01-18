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
def _create_write_item_for_shard(fqn: str, sharded_tensor: ShardedTensor, shard_md: ShardMetadata) -> WriteItem:
    offsets = torch.Size(shard_md.shard_offsets)
    return WriteItem(index=MetadataIndex(fqn, offsets), type=WriteItemType.SHARD, tensor_data=_sharded_tensor_metadata(sharded_tensor, shard_md))