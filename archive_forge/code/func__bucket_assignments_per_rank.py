import collections
import copy
import enum
import inspect
import io
import logging
from itertools import chain
from typing import Any, Callable, Dict, List, Optional, Set, Type, Union
import torch
import torch.distributed as dist
from torch.distributed.algorithms.join import Join, Joinable, JoinHook
from torch.distributed.optim.utils import functional_optim_map
from torch.optim import Optimizer
@property
def _bucket_assignments_per_rank(self) -> List[Dict[int, _DDPBucketAssignment]]:
    """
        Return DDP bucket parameters assigned per rank.

        :class:`list` of length world size consisting of :class:`dict` s
        mapping bucket indices to :class:`_DDPBucketAssignment` s for each
        rank.
        """
    assert self._overlap_with_ddp, '`_bucket_assignments_per_rank` only be used if `overlap_with_ddp=True`'
    if len(self._bucket_assignments_per_rank_cache) > 0:
        return self._bucket_assignments_per_rank_cache
    overlap_info = self._overlap_info
    assert overlap_info.status == _OverlapStatus.INITIALIZED
    self._bucket_assignments_per_rank_cache = [{} for _ in range(self.world_size)]
    params_per_bucket = overlap_info.params_per_bucket
    if overlap_info.shard_buckets:
        assert overlap_info.total_size is not None, '`total_size` was not computed'
        threshold = overlap_info.total_size / self.world_size
        size_per_rank = [0 for _ in range(self.world_size)]
    num_buckets = len(params_per_bucket)
    overlap_info.assigned_ranks_per_bucket = [set() for _ in range(num_buckets)]
    assigned_ranks_per_bucket = overlap_info.assigned_ranks_per_bucket
    if not overlap_info.shard_buckets:
        for bucket_index, bucket_params in enumerate(params_per_bucket):
            assert len(bucket_params) > 0, 'Empty bucket'
            assigned_rank = self._get_assigned_rank(bucket_index)
            self._assign_bucket_subset_to_rank(bucket_index, bucket_params, 0, assigned_rank, assigned_ranks_per_bucket)
    else:
        params_per_bucket_enum = sorted(enumerate(params_per_bucket), key=lambda x: sum((p.numel() for p in x[1])))
        for bucket_index, bucket_params in params_per_bucket_enum:
            assert len(bucket_params) > 0, 'Empty bucket'
            bucket_offset = 0
            assignment_size = 0
            for param_index, param in enumerate(bucket_params):
                param_numel = param.numel()
                if assignment_size + param_numel >= threshold and param_index > bucket_offset:
                    assigned_rank = self._get_min_index(size_per_rank, assigned_ranks_per_bucket[bucket_index])
                    self._assign_bucket_subset_to_rank(bucket_index, bucket_params[bucket_offset:param_index], bucket_offset, assigned_rank, assigned_ranks_per_bucket)
                    size_per_rank[assigned_rank] += assignment_size
                    bucket_offset = param_index
                    assignment_size = 0
                assignment_size += param_numel
            assigned_rank = self._get_min_index(size_per_rank, assigned_ranks_per_bucket[bucket_index])
            self._assign_bucket_subset_to_rank(bucket_index, bucket_params[bucket_offset:], bucket_offset, assigned_rank, assigned_ranks_per_bucket)
            size_per_rank[assigned_rank] += assignment_size
    return self._bucket_assignments_per_rank_cache