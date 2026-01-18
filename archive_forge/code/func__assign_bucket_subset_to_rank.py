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
def _assign_bucket_subset_to_rank(self, bucket_index: int, bucket_params: List[torch.Tensor], bucket_offset: int, assigned_rank: int, assigned_ranks_per_bucket: List[Set[int]]) -> None:
    """
        Assign ``bucket_params`` to the rank with the least size assigned so far and collects relevant information.

        The model parameters given by ``bucket_params`` represents a (possibly non-strict)
        subset of the parameters corresponding to a :class:`DistributedDataParallel` bucket.

        Arguments:
            bucket_index (int): index of the :class:`DistributedDataParallel`
                gradient bucket.
            bucket_params (List[torch.Tensor]): subset of the parameters
                corresponding to the bucket to assign.
            bucket_offset (int): offset giving the index of the first element
                in ``bucket_params`` in the bucket's full parameter list.
            assigned_rank (int): group rank to assign to.
            assigned_ranks_per_bucket (List[Set[int]]): :class:`set` of group ranks
                assigned to each bucket.
        """
    overlap_info = self._overlap_info
    if len(bucket_params) == 0:
        raise ValueError('Empty bucket assignment')
    params_per_rank = overlap_info.params_per_rank
    offsets = overlap_info.offsets
    self._bucket_assignments_per_rank_cache[assigned_rank][bucket_index] = _DDPBucketAssignment(bucket_index, bucket_params, bucket_offset)
    if self.global_rank == assigned_rank:
        offsets[bucket_index] = len(params_per_rank[assigned_rank])
    params_per_rank[assigned_rank].extend(bucket_params)
    assigned_ranks_per_bucket[bucket_index].add(assigned_rank)
    self._overlap_info.num_bucket_assignments += 1