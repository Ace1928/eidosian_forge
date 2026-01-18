import logging
import warnings
import weakref
import torch
import torch.distributed as dist
import torch.distributed.distributed_c10d as c10d
from typing import List, Optional, cast
def _reduce_scatter_tensor_coalesced_fallback(output_tensors, input_tensors, op, group, async_op=False):
    work_list = []
    for shard, out_tensor in zip(input_tensors, output_tensors):
        work = c10d.reduce_scatter_tensor(out_tensor, shard, op=op, group=group, async_op=async_op)
        work_list.append(work)
    return work_list