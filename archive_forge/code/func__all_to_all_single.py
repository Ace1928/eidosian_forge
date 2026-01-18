import logging
import warnings
import weakref
import torch
import torch.distributed as dist
import torch.distributed.distributed_c10d as c10d
from typing import List, Optional, cast
def _all_to_all_single(input: torch.Tensor, output_split_sizes: Optional[List[int]], input_split_sizes: Optional[List[int]], tag: str, ranks: List[int], group_size: int):
    group = c10d._find_or_create_pg_by_ranks_and_tag(tag, ranks, group_size)
    if output_split_sizes is not None:
        torch._check(input.dim() >= 1, lambda: f'Expected input to have at least 1 dim but got {input.dim()} dim')
        out_size = list(input.size())
        out_size[0] = sum(output_split_sizes)
        out_tensor = input.new_empty(out_size)
    else:
        out_tensor = input.new_empty(input.size())
    work = c10d.all_to_all_single(out_tensor, input, output_split_sizes=output_split_sizes, input_split_sizes=input_split_sizes, group=group, async_op=True)
    _register_tensor_work(out_tensor, work)
    return out_tensor