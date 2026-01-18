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
class _OverlapInfo:
    """
    Information needed by :class:`ZeroRedundancyOptimizer` to overlap with :class:`DistributedDataParallel`.

    Arguments:
        world_size (int): world size of the process group being used.

    Attributes:
        shard_buckets (bool): if ``True``, then the assignment of each
            :class:`DistributedDataParallel` bucket is partitioned across
            possibly multiple :class:`ZeroRedundancyOptimizer` instances (i.e.
            across possibly multiple ranks) to approximate uniformity following
            a threshold given by the total parameter size divided by the world
            size; if ``False``, then each bucket is wholly assigned to a single
            :class:`ZeroRedundancyOptimizer` instance (i.e. to a single rank);
            this should be set to the value passed into the hook constructor.
        status (_OverlapStatus): current status; see :class:`_OverlapStatus`
            for more information.
        params_per_bucket (List[List[torch.Tensor]]): ``params_per_bucket[i]``
            gives the model parameters in the ``i``th bucket.
        params_per_rank (List[List[torch.Tensor]]): ``params_per_rank[i]``
            gives the model parameters assigned to the ``i``th rank, where the
            parameters are grouped by increasing bucket indices.
        offsets (Dict[int, int]): maps from bucket index to the offset in
            ``self.params_per_rank[rank]`` giving the index of the first
            parameter in that bucket, where ``rank`` is this process's own
            rank; the keys of this :class:`dict` are the bucket indices
            assigned to this rank.
        num_bucket_assignments (int): total number of bucket assignments across
            all ranks; this is equal to the number of
            :class:`DistributedDataParallel` gradient buckets if
            ``shard_buckets=False`` and possibly greater otherwise.
        total_size (int, optional): total size of all buckets (i.e. sum of
            ``param.numel()`` for all ``param`` across all buckets) if
            ``shard_buckets=True``; otherwise, ``None``.
        broadcast_handles (List[Work]): :class:`list` of async work handles for
            the parameter broadcasts.
        bucket_index_to_future (Dict[int, torch.futures.Future]):
            :class:`dict` mapping bucket index to the corresponding all-reduce
            future.
        bucket_index_to_bucket (Dict[int, dist.GradBucket]): :class:`dict`
            mapping bucket index to the corresponding bucket.
        bucket_indices_seen (List[int]): :class:`list` of the bucket indices
            seen on this iteration.
    """

    def __init__(self, world_size) -> None:
        self.status: _OverlapStatus = _OverlapStatus.UNINITIALIZED
        self.shard_buckets: bool = False
        self.params_per_bucket: List[List[torch.Tensor]] = []
        self.params_per_rank: List[List[torch.Tensor]] = [[] for _ in range(world_size)]
        self.offsets: Dict[int, int] = {}
        self.assigned_ranks_per_bucket: List[Set[int]] = []
        self.num_bucket_assignments: int = 0
        self.total_size: Optional[int] = None
        self.broadcast_handles: List[Any] = []
        self.bucket_indices_seen: List[int] = []
        self.bucket_index_to_future: Dict[int, torch.futures.Future] = {}
        self.bucket_index_to_bucket: Dict[int, dist.GradBucket] = {}

    def wait_for_broadcasts(self) -> None:
        """
        Wait for all parameter broadcasts.

        This function should be called once all broadcasts have been scheduled,
        meaning ``self.broadcast_handles`` is filled. This clears ``self.broadcast_handles``
        in preparation for the next iteration.
        """
        assert len(self.broadcast_handles) == self.num_bucket_assignments, f'Missing at least one broadcast handle on rank {dist.get_rank()}'
        _ = [x.wait() for x in self.broadcast_handles]
        self.broadcast_handles.clear()

    def clear_per_iter_info(self) -> None:
        """
        Clear the data structures that are modified per-iteration.

        This function should be called at the end of an iteration.
        """
        self.bucket_indices_seen.clear()
        self.bucket_index_to_future.clear()
        self.bucket_index_to_bucket.clear()