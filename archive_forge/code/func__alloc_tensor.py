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
def _alloc_tensor(props: TensorProperties, size: Sequence[int], device_type: str='cuda') -> torch.Tensor:
    return torch.empty(size=size, dtype=props.dtype, layout=props.layout, requires_grad=props.requires_grad, pin_memory=props.pin_memory, device=cast(torch.device, _get_device_module(device_type).current_device()))