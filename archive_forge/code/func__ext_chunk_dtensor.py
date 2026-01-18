from abc import ABC, abstractmethod
from typing import Any, List, Optional, Tuple
import torch
import torch.distributed as dist
from torch.distributed._shard.sharded_tensor.api import ShardedTensor
from torch.distributed._shard.sharded_tensor.shard import Shard
from torch.distributed._tensor import DeviceMesh, DTensor
from torch.distributed.fsdp._shard_utils import (
def _ext_chunk_dtensor(tensor: torch.Tensor, rank: int, device_mesh: DeviceMesh, fsdp_extension: Optional[FSDPExtensions]=None) -> torch.Tensor:
    chunk_dtensor_fn = fsdp_extension.chunk_dtensor if fsdp_extension is not None else _create_chunk_dtensor
    return chunk_dtensor_fn(tensor, rank, device_mesh)