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
def _create_write_items(fqn: str, object: Any) -> List[WriteItem]:
    if isinstance(object, DTensor):
        return [_create_write_items_for_dtensor(fqn, object)]
    elif isinstance(object, ShardedTensor):
        return [_create_write_item_for_shard(fqn, object, shard.metadata) for shard in object.local_shards()]
    elif isinstance(object, torch.Tensor):
        return [_create_write_item_for_tensor(fqn, object)]
    else:
        return [_create_write_item_for_bytesio(fqn, object)]