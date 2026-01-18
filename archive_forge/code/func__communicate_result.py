import torch
import torch.distributed as dist
import torch.distributed.distributed_c10d as distributed_c10d
from torch.distributed._shard.sharded_tensor import (
def _communicate_result(result, pg):
    if result:
        result_tensor = torch.ones(1, device=torch.device(torch.cuda.current_device()))
    else:
        result_tensor = torch.zeros(1, device=torch.device(torch.cuda.current_device()))
    dist.all_reduce(result_tensor, group=pg)
    expected_result = torch.ones(1, device=torch.device(torch.cuda.current_device())) * dist.get_world_size(pg)
    return torch.equal(result_tensor, expected_result)