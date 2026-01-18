import weakref
from typing import Any, Callable, List, Optional
import torch
import torch.distributed as dist
from torch.distributed.optim import ZeroRedundancyOptimizer
from torch.distributed.optim.zero_redundancy_optimizer import (
from torch.nn.parallel.distributed import DistributedDataParallel
def _broadcast_bucket(bucket_index: int, zero: ZeroRedundancyOptimizer):
    """
    Broadcasts a bucket's parameters.

    Arguments:
        bucket_index (int): the index of the bucket corresponding to the
            parameters to broadcast.
        zero (ZeroRedundancyOptimizer): the calling process's
            :class:`ZeroRedundancyOptimizer` instance.
    """
    overlap_info = zero._overlap_info
    assert len(overlap_info.assigned_ranks_per_bucket) > bucket_index, '`assigned_ranks_per_bucket` is not fully constructed'
    assigned_ranks = sorted(overlap_info.assigned_ranks_per_bucket[bucket_index])
    assert len(assigned_ranks) > 0, f'Bucket {bucket_index} should be assigned to at least one rank'
    for assigned_rank in assigned_ranks:
        bucket_assignments = zero._bucket_assignments_per_rank[assigned_rank]
        if bucket_index in bucket_assignments:
            overlap_info.broadcast_handles.append(dist.broadcast(bucket_assignments[bucket_index].tensor, src=dist.get_global_rank(zero.process_group, assigned_rank), group=zero.process_group, async_op=True))