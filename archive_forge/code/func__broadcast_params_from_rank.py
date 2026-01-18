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
def _broadcast_params_from_rank(self, rank: int):
    """
        Broadcast the shard of parameters from a given rank to all other ranks asynchronously.

        Arguments:
            rank (int): the source rank.

        Returns:
            A :class:`list` of async work handles for the ``broadcast()`` s
            performed to synchronize the parameters.
        """
    assert not self._overlap_with_ddp, '`_broadcast_params_from_rank()` should not be used if `overlap_with_ddp=True`; instead, the broadcasting should happen in the DDP communication hook'
    handles = []
    if self.parameters_as_bucket_view:
        for dev_i_buckets in self._buckets:
            bucket = dev_i_buckets[rank]
            global_rank = dist.distributed_c10d.get_global_rank(self.process_group, rank)
            handles.append(dist.broadcast(tensor=bucket, src=global_rank, group=self.process_group, async_op=True))
    else:
        param_groups = self._partition_parameters()[rank]
        global_rank = dist.distributed_c10d.get_global_rank(self.process_group, rank)
        for param_group in param_groups:
            for param in param_group['params']:
                handles.append(dist.broadcast(tensor=param.data, src=global_rank, group=self.process_group, async_op=True))
    return handles