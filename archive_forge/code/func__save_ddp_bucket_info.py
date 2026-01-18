import weakref
from typing import Any, Callable, List, Optional
import torch
import torch.distributed as dist
from torch.distributed.optim import ZeroRedundancyOptimizer
from torch.distributed.optim.zero_redundancy_optimizer import (
from torch.nn.parallel.distributed import DistributedDataParallel
def _save_ddp_bucket_info(bucket: dist.GradBucket, zero: ZeroRedundancyOptimizer):
    """
    Saves :class:`DistributedDataParallel` gradient bucket information for the
    :class:`ZeroRedundancyOptimizer` instance ``zero`` to use when overlapping.
    In particular, this function is meant to be called upon seeing each
    gradient bucket, meaning it does not save or compute any global
    information.

    Arguments:
        bucket (dist.GradBucket): the current gradient bucket.
        zero (ZeroRedundancyOptimizer): the calling process's
            :class:`ZeroRedundancyOptimizer` instance.
    """
    overlap_info = zero._overlap_info
    bucket_params = bucket.parameters()
    assert len(bucket_params) > 0, 'Empty bucket'
    overlap_info.params_per_bucket.append(bucket_params)
    if overlap_info.shard_buckets:
        bucket_size = 0
        for param in bucket_params:
            bucket_size += param.numel()
        assert overlap_info.total_size is not None
        overlap_info.total_size += bucket_size