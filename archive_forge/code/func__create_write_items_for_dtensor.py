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
def _create_write_items_for_dtensor(fqn: str, tensor: DTensor) -> WriteItem:
    sizes, offsets = compute_local_shape_and_global_offset(tensor.shape, tensor.device_mesh, tensor.placements)
    sizes, offsets = (torch.Size(sizes), torch.Size(offsets))
    return WriteItem(index=MetadataIndex(fqn, offsets), type=WriteItemType.SHARD, tensor_data=TensorWriteData(chunk=ChunkStorageMetadata(offsets=offsets, sizes=sizes), properties=TensorProperties.create_from_tensor(tensor.to_local()), size=tensor.size()))