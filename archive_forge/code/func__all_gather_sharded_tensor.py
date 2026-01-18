import math
from typing import Any, Callable, Dict, Optional, Tuple, TYPE_CHECKING
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed._functional_collectives import AsyncCollectiveTensor
def _all_gather_sharded_tensor(sharded_tensor: ShardedTensor, pg: Optional[dist.ProcessGroup]=None, device: Optional[torch.device]=None) -> torch.Tensor:
    if pg is None:
        pg = distributed_c10d._get_default_group()
    world_size = dist.get_world_size(pg)
    shards = sharded_tensor.local_shards()
    dim_0_size = sharded_tensor.size()[0]
    tensor_numel = sharded_tensor.size().numel()
    chunk_size = math.ceil(dim_0_size / world_size) * tensor_numel // dim_0_size
    pg_device = distributed_c10d._get_pg_default_device(pg) if device is None else device
    if shards:
        local_tensor = shards[0].tensor.flatten()
        if local_tensor.device.type != pg_device.type:
            local_tensor = local_tensor.to(pg_device)
        num_padding = chunk_size - local_tensor.numel()
        if num_padding > 0:
            local_tensor = F.pad(local_tensor, [0, num_padding])
    else:
        local_tensor = torch.zeros(chunk_size, dtype=sharded_tensor.dtype, device=pg_device)
    tensor = torch.empty(chunk_size * world_size, dtype=local_tensor.dtype, device=pg_device)
    dist.all_gather_into_tensor(tensor, local_tensor, group=pg)
    tensor = tensor.narrow(0, 0, tensor_numel).reshape(sharded_tensor.size())
    return tensor