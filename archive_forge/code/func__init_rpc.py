from __future__ import annotations  # type: ignore[attr-defined]
from dataclasses import dataclass
from typing import (
import copy
import warnings
from functools import reduce
import weakref
import threading
import torch
import torch.distributed as dist
from torch.distributed import rpc
from torch.distributed import distributed_c10d
from torch.distributed._shard.metadata import ShardMetadata
import torch.distributed._shard.sharding_spec as shard_spec
from torch.distributed._shard.sharding_spec.api import (
from torch.distributed._shard.sharding_spec._internals import (
from torch.distributed._shard._utils import (
from .metadata import TensorProperties, ShardedTensorMetadata
from .shard import Shard
from .reshard import reshuffle_local_shard, reshard_local_shard
from .utils import (
from torch.distributed.remote_device import _remote_device
from torch.utils import _pytree as pytree
def _init_rpc(self):
    pg_rank = dist.get_rank()
    rpc_rank = rpc.get_worker_info().id
    if pg_rank != rpc_rank:
        raise ValueError(f'Default ProcessGroup and RPC ranks must be the same for ShardedTensor, found process group rank: {pg_rank} and RPC rank: {rpc_rank}')
    self._remote_shards = {}
    worker_infos = rpc._get_current_rpc_agent().get_worker_infos()
    rank_to_name = {}
    name_to_rank = {}
    for worker_info in worker_infos:
        rank_to_name[worker_info.id] = worker_info.name
        name_to_rank[worker_info.name] = worker_info.id
    all_tensor_ids = rpc.api._all_gather(self._sharded_tensor_id)
    futs = []
    rpc_rank = rpc.get_worker_info().id
    for rank in range(dist.get_world_size()):
        if rank == dist.get_rank():
            continue
        if len(self.local_shards()) != 0:
            rrefs: List[rpc.RRef[Shard]] = [rpc.RRef(shard) for shard in self.local_shards()]
            fut = rpc.rpc_async(rank, _register_remote_shards, args=(all_tensor_ids[rank_to_name[rank]], rrefs, rpc_rank))
            futs.append(fut)
    torch.futures.wait_all(futs)
    rpc.api._all_gather(None)