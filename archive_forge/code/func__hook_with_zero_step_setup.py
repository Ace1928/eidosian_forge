import weakref
from typing import Any, Callable, List, Optional
import torch
import torch.distributed as dist
from torch.distributed.optim import ZeroRedundancyOptimizer
from torch.distributed.optim.zero_redundancy_optimizer import (
from torch.nn.parallel.distributed import DistributedDataParallel
def _hook_with_zero_step_setup(ddp_ref: weakref.ReferenceType, zero: ZeroRedundancyOptimizer, bucket: dist.GradBucket):
    """
    Encapsulates the setup logic for :func:`hook_with_zero_step` and
    :func:`hook_with_zero_step_interleaved`, meaning the logic to run in the
    hook before the backward pass and optimizer step can actually be
    overlapped. This is factored out since it is common to both
    :func:`hook_with_zero_step` and :func:`hook_with_zero_step_interleaved`.

    Arguments:
        ddp_ref (weakref.ReferenceType): weak reference to the process's
            :class:`DistributedDataParallel` instance.
        zero (ZeroRedundancyOptimizer): the calling process's
            :class:`ZeroRedundancyOptimizer` instance.
        bucket (dist.GradBucket): the current gradient bucket.
    """
    if not ddp_ref()._has_rebuilt_buckets:
        assert zero._overlap_info.status == _OverlapStatus.UNINITIALIZED
        return
    bucket_index = bucket.index()
    overlap_info = zero._overlap_info
    if overlap_info.status == _OverlapStatus.UNINITIALIZED:
        overlap_info.status = _OverlapStatus.DDP_HAS_REBUILT_BUCKETS
    if overlap_info.status == _OverlapStatus.DDP_HAS_REBUILT_BUCKETS:
        if bucket_index == 0 and len(overlap_info.params_per_bucket) > 0:
            zero._init_zero_for_overlap()
        else:
            _save_ddp_bucket_info(bucket, zero)