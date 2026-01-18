import collections
import enum
from typing import cast, Dict, List, Set, Tuple
import torch
import torch.distributed as dist
from ._utils import _group_membership_management, _update_group_membership
from . import api
from . import constants as rpc_constants
def _init_process_group(store, rank, world_size):
    process_group_timeout = rpc_constants.DEFAULT_PROCESS_GROUP_TIMEOUT
    group = dist.ProcessGroupGloo(store, rank, world_size, process_group_timeout)
    assert group is not None, 'Failed to initialize default ProcessGroup.'
    if rank != -1 and rank != group.rank():
        raise RuntimeError(f"rank argument {rank} doesn't match pg rank {group.rank()}")
    if world_size != -1 and world_size != group.size():
        raise RuntimeError(f"world_size argument {world_size} doesn't match pg size {group.size()}")
    return group