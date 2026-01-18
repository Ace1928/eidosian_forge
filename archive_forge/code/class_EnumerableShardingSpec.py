from abc import ABC, abstractmethod
from dataclasses import dataclass
import functools
from typing import Callable, Dict, List, TYPE_CHECKING
import torch
from ._internals import (
from torch.distributed._shard.metadata import ShardMetadata
import torch.distributed._shard.sharded_tensor.metadata as sharded_tensor_meta
from torch.distributed._shard.op_registry_utils import _decorator_func
@dataclass
class EnumerableShardingSpec(ShardingSpec):
    """
    This is a type of PlacementSpec that allows users to specify a generic
    sharding scheme by enumerating exactly how each shard is laid out.

    Args:
        shards(List[ShardMetadata]): List of :class:`ShardMetadata` objects representing
            each shard. Note that none of the shards should overlap.
    """
    shards: List[ShardMetadata]

    def __post_init__(self):
        if len(self.shards) == 0:
            raise ValueError(f'Empty shard list provided: {self.shards}')
        rank = -1
        for shard in self.shards:
            if rank != -1 and rank != len(shard.shard_offsets):
                raise ValueError(f'Found inconsistent ranks for shards: {rank} and {len(shard.shard_offsets)}')
            rank = len(shard.shard_offsets)
        validate_non_overlapping_shards_metadata(self.shards)

    def build_metadata(self, tensor_sizes: torch.Size, tensor_properties: sharded_tensor_meta.TensorProperties) -> sharded_tensor_meta.ShardedTensorMetadata:
        check_tensor(self.shards, tensor_sizes)
        return sharded_tensor_meta.ShardedTensorMetadata(self.shards, tensor_sizes, tensor_properties)

    def shard(self, tensor: torch.Tensor, src_rank: int=0, process_group=None) -> 'ShardedTensor':
        raise NotImplementedError('EnumerableShardingSpec.shard not implemented yet!')