from abc import ABC, abstractmethod
from typing import Any, List, Optional, Tuple
import torch
import torch.distributed as dist
from torch.distributed._shard.sharded_tensor.api import ShardedTensor
from torch.distributed._shard.sharded_tensor.shard import Shard
from torch.distributed._tensor import DeviceMesh, DTensor
from torch.distributed.fsdp._shard_utils import (
def _ext_pre_load_state_dict_transform(tensor: torch.Tensor, fsdp_extension: Optional[FSDPExtensions]=None) -> Tuple[torch.Tensor, List[Shard]]:
    if fsdp_extension is not None:
        return fsdp_extension.pre_load_state_dict_transform(tensor)
    assert type(tensor) is ShardedTensor
    shards = tensor.local_shards()
    return (tensor, shards)