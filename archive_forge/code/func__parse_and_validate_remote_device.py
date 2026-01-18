import collections.abc
import copy
from typing import Optional, List, Sequence
import torch
from torch.distributed import distributed_c10d
from torch.distributed import rpc
from torch.distributed._shard.sharding_spec._internals import (
from torch.distributed._shard.metadata import ShardMetadata
from .metadata import TensorProperties, ShardedTensorMetadata
from .shard import Shard
def _parse_and_validate_remote_device(pg, remote_device):
    if remote_device is None:
        raise ValueError('remote device is None')
    worker_name = remote_device.worker_name()
    rank = remote_device.rank()
    device = remote_device.device()
    if not distributed_c10d._rank_not_in_group(pg):
        if rank is not None and (rank < 0 or rank >= distributed_c10d.get_world_size(pg)):
            raise ValueError(f'Invalid rank: {rank}')
    if worker_name is not None:
        if not rpc._is_current_rpc_agent_set():
            raise RuntimeError(f'RPC framework needs to be initialized for using worker names: {worker_name}')
        workers = rpc._get_current_rpc_agent().get_worker_infos()
        for worker in workers:
            if worker.name == worker_name:
                return (worker.id, device)
        raise ValueError(f'Invalid worker name: {worker_name}')
    return (rank, device)