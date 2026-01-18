import contextlib
import logging
import os
import time
from contextlib import nullcontext
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Iterator, List, Optional, Sized, Union
import torch
import torch.nn.functional as F
from lightning_utilities.core.imports import package_available
from torch import Tensor
from torch.utils.data import Dataset, DistributedSampler, Sampler
from typing_extensions import override
from lightning_fabric.utilities.cloud_io import _is_local_file_protocol
from lightning_fabric.utilities.data import _num_cpus_available
from lightning_fabric.utilities.rank_zero import rank_zero_info
from lightning_fabric.utilities.types import _PATH, ReduceOp
def _sync_ddp(result: Tensor, group: Optional[Any]=None, reduce_op: Optional[Union[ReduceOp, str]]=None) -> Tensor:
    """Reduces a tensor across several distributed processes.

    This operation is performed in-place, meaning the result will be placed back into the input tensor on all processes.

    Args:
        result: The value to sync and reduce (typically tensor or number)
        group: The process group to gather results from. Defaults to all processes (world)
        reduce_op: The reduction operation. Defaults to sum.
            Can also be a string of 'avg', 'mean' to calculate the mean during reduction.

    Return:
        The reduced value.

    """
    divide_by_world_size = False
    group = torch.distributed.group.WORLD if group is None else group
    op: Optional[ReduceOp]
    if isinstance(reduce_op, str):
        reduce_op = 'avg' if reduce_op == 'mean' else reduce_op
        if reduce_op.lower() == 'avg' and torch.distributed.get_backend(group) == 'gloo':
            op = ReduceOp.SUM
            divide_by_world_size = True
        else:
            op = getattr(ReduceOp, reduce_op.upper())
    else:
        op = reduce_op
    if package_available('habana_frameworks') and os.environ.get('HCCL_DISTRIBUTED_BACKEND') == '1' and (result.type() in ('torch.LongTensor', 'torch.hpu.LongTensor')):
        rank_zero_info('Long tensor unsupported on HPU, casting to float')
        result = result.float()
    torch.distributed.barrier(group=group)
    torch.distributed.all_reduce(result, op=op, group=group, async_op=False)
    world_size = torch.distributed.get_world_size(group)
    if not divide_by_world_size:
        return result
    if not torch.is_floating_point(result):
        return result.copy_(result / world_size)
    return result.div_(world_size)