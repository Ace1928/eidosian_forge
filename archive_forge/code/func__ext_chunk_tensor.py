from abc import ABC, abstractmethod
from typing import Any, List, Optional, Tuple
import torch
import torch.distributed as dist
from torch.distributed._shard.sharded_tensor.api import ShardedTensor
from torch.distributed._shard.sharded_tensor.shard import Shard
from torch.distributed._tensor import DeviceMesh, DTensor
from torch.distributed.fsdp._shard_utils import (
def _ext_chunk_tensor(tensor: torch.Tensor, rank: int, world_size: int, num_devices_per_node: int, pg: dist.ProcessGroup, fsdp_extension: Optional[FSDPExtensions]=None) -> torch.Tensor:
    chunk_tensor_fn = fsdp_extension.chunk_tensor if fsdp_extension is not None else _create_chunk_sharded_tensor
    return chunk_tensor_fn(tensor, rank, world_size, num_devices_per_node, pg)