import logging
import warnings
import weakref
import torch
import torch.distributed as dist
import torch.distributed.distributed_c10d as c10d
from typing import List, Optional, cast
def _reduce_scatter_tensor(input: torch.Tensor, reduceOp: str, tag: str, ranks: List[int], group_size: int):
    group = c10d._find_or_create_pg_by_ranks_and_tag(tag, ranks, group_size)
    assert group is not None
    op = _str_to_reduce_op(reduceOp)
    if dist.get_backend(group) == dist.Backend.GLOO or input.is_cpu:
        logger.warning('ProcessGroupGloo does not support reduce_scatter, falling back with all reduce!')
        reduction_input = input.clone()
        group_rank = dist.get_rank(group)
        work = dist.all_reduce(reduction_input, op=op, group=group, async_op=True)
        out_tensor = reduction_input.chunk(group_size, dim=0)[group_rank]
        _register_tensor_work(out_tensor, work)
    else:
        out_size = list(input.size())
        out_size[0] //= group_size
        out_tensor = input.new_empty(out_size)
        work = dist.reduce_scatter_tensor(out_tensor, input, op=op, group=group, async_op=True)
        _register_tensor_work(out_tensor, work)
    return out_tensor