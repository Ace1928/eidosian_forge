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
@classmethod
def _init_from_local_shards(cls, local_shards: List[Shard], *global_size, process_group=None, init_rrefs=False):
    process_group = process_group if process_group is not None else distributed_c10d._get_default_group()
    current_rank = dist.get_rank(process_group)
    world_size = dist.get_world_size(process_group)
    local_sharded_tensor_metadata: Optional[ShardedTensorMetadata] = None
    global_tensor_size = _flatten_tensor_size(global_size)
    if len(local_shards) > 0:
        local_sharded_tensor_metadata = build_metadata_from_local_shards(local_shards, global_tensor_size, current_rank, process_group)
    gathered_metadatas: List[Optional[ShardedTensorMetadata]] = []
    if world_size > 1:
        gathered_metadatas = [None for _ in range(world_size)]
        dist.all_gather_object(gathered_metadatas, local_sharded_tensor_metadata, group=process_group)
    else:
        gathered_metadatas = [local_sharded_tensor_metadata]
    global_sharded_tensor_metadata = build_global_metadata(gathered_metadatas)
    tensor_properties = global_sharded_tensor_metadata.tensor_properties
    spec = shard_spec._infer_sharding_spec_from_shards_metadata(global_sharded_tensor_metadata.shards_metadata)
    sharded_tensor = cls.__new__(cls, spec, global_sharded_tensor_metadata.size, dtype=tensor_properties.dtype, layout=tensor_properties.layout, pin_memory=tensor_properties.pin_memory, requires_grad=tensor_properties.requires_grad)
    sharded_tensor._prepare_init(process_group=process_group, init_rrefs=init_rrefs)
    sharded_tensor._local_shards = local_shards
    sharded_tensor._post_init()
    return sharded_tensor