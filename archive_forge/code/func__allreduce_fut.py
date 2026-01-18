from typing import Any, Callable
import torch
import torch.distributed as dist
def _allreduce_fut(process_group: dist.ProcessGroup, tensor: torch.Tensor) -> torch.futures.Future[torch.Tensor]:
    """Averages the input gradient tensor by allreduce and returns a future."""
    group_to_use = process_group if process_group is not None else dist.group.WORLD
    tensor.div_(group_to_use.size())
    return dist.all_reduce(tensor, group=group_to_use, async_op=True).get_future().then(lambda fut: fut.value()[0])