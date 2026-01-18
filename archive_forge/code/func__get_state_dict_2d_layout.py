import dataclasses
from typing import Dict, List, Optional, Sequence, Tuple, Union, cast
from torch.distributed.checkpoint.planner import LoadPlan
import torch
import torch.distributed as dist
from torch.distributed._shard.sharded_tensor.api import ShardedTensor
from torch.distributed._shard.sharded_tensor.metadata import TensorProperties
from torch.distributed._shard.sharded_tensor.shard import Shard
from torch.distributed._shard.sharding_spec.chunk_sharding_spec import (
import torch.distributed.checkpoint as dist_cp
from torch.distributed.checkpoint.metadata import (
from torch.distributed.distributed_c10d import _get_default_group
from torch.distributed.fsdp._shard_utils import _create_chunk_sharded_tensor
from torch.distributed.checkpoint.planner_helpers import (
from torch.distributed.remote_device import _remote_device
from torch.distributed._tensor import DTensor
from torch.distributed.checkpoint.default_planner import (
from torch.distributed.checkpoint.planner import LoadPlanner
from torch.distributed.checkpoint._nested_dict import unflatten_state_dict
from torch.distributed.checkpoint.utils import (
from torch._utils import _get_device_module
def _get_state_dict_2d_layout(state_dict: STATE_DICT_TYPE) -> Tuple[STATE_DICT_2D_LAYOUT, Optional[dist.ProcessGroup]]:
    """
    Load the right TP slice of the optimizer state.

    This is not easy since the per-tensor slicing can't be inferred from checkpoint metadata.
    We take advantage of the model state_dict producing a sliced ST to figure out what we need to load.
    This is pretty fragile and it might be easier for FSDP to compute this info for us.
    Returns a dictionary where keys are the same of the state_dict and the value is a tuple of
    (offset, size) for the current rank TP slice.
    N.B. The state_dict *MUST* come from FSDP.sharded_state_dict.
    """
    specs: STATE_DICT_2D_LAYOUT = {}
    dp_pg: Optional[dist.ProcessGroup] = None
    for key, value in state_dict.items():
        specs[key] = (None, value.size())
        if _is_nested_tensor(value):
            assert len(value.local_shards()) == 1, 'Cannot handle ST with multiple shards'
            assert isinstance(value, ShardedTensor), 'Can only handle nested ShardedTensor'
            shard = value.local_shards()[0]
            specs[key] = (shard.metadata.shard_offsets, shard.metadata.shard_sizes)
            dp_pg = shard.tensor._process_group
    return (specs, dp_pg)