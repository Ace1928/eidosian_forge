import copy
import torch.distributed as dist
from torch.distributed.remote_device import _remote_device
from torch.distributed.checkpoint.metadata import (
from torch.distributed._shard.sharded_tensor import (
from torch.distributed._shard.sharded_tensor.metadata import (
from ._traverse import (
from .utils import _element_wise_add, _normalize_device_info

    Transform ``state_dict`` by flattening all nested ShardedTensor instances found.

    The resulting ShardedTensor instances are only correct regarding the local shard and
    MUST not be used for any other purpose but checkpointing, as no operator will work with them.

    This function should be used in conjunction with a state_dict produced by FSDP's
    StateDictType.SHARDED_STATE_DICT methods.
    