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
class _ReaderWithOffset(DefaultLoadPlanner):
    translation: Dict[MetadataIndex, MetadataIndex]
    state_dict: STATE_DICT_TYPE
    metadata: Metadata

    def __init__(self, fqn_to_offset: Dict[str, Sequence[int]]) -> None:
        super().__init__()
        self.fqn_to_offset = fqn_to_offset
        self.metadata = Metadata({})
        self.state_dict = {}
        self.translation = {}

    def create_local_plan(self) -> LoadPlan:
        requests = []
        self.translation = {}
        for fqn, obj in self.state_dict.items():
            md = self.metadata.state_dict_metadata[fqn]
            if not isinstance(obj, ShardedTensor):
                requests += _create_read_items(fqn, md, obj)
                continue
            if fqn not in self.fqn_to_offset:
                requests += _create_read_items(fqn, md, obj)
                continue
            offset = self.fqn_to_offset[fqn]
            assert len(obj.local_shards()) == 1
            original_shard = obj.local_shards()[0]
            local_chunks = [ChunkStorageMetadata(offsets=torch.Size(_element_wise_add(original_shard.metadata.shard_offsets, offset)), sizes=torch.Size(original_shard.metadata.shard_sizes))]
            reqs = create_read_items_for_chunk_list(fqn, cast(TensorStorageMetadata, md), local_chunks)
            for ri in reqs:
                assert ri.dest_index.offset is not None
                original_offset = _element_wise_sub(ri.dest_index.offset, offset)
                original_index = dataclasses.replace(ri.dest_index, offset=torch.Size(original_offset))
                self.translation[ri.dest_index] = original_index
            requests += reqs
        return LoadPlan(requests)

    def lookup_tensor(self, index: MetadataIndex) -> torch.Tensor:
        return super().lookup_tensor(self.translation.get(index, index))