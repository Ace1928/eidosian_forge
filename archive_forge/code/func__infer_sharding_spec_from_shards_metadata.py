from abc import ABC, abstractmethod
from dataclasses import dataclass
import functools
from typing import Callable, Dict, List, TYPE_CHECKING
import torch
from ._internals import (
from torch.distributed._shard.metadata import ShardMetadata
import torch.distributed._shard.sharded_tensor.metadata as sharded_tensor_meta
from torch.distributed._shard.op_registry_utils import _decorator_func
def _infer_sharding_spec_from_shards_metadata(shards_metadata):
    """
    Infer the sharding spec from the metadata of each shard of a ShardedTensor.
    If the tensor is sharded only on one dimension, we can then verify whether it's
    a ChunkShardingSpec or not. The way to verify it is to first get the total length
    and perform a chunk sharding with the given placements to see if we can have the
    same chunk size as the given shards_metadata. If not, we assume it's enum sharded.

    Args:
        shards_metadata (List[ShardMetadata]): List of Metadata of local shards.

    Returns:
        A :class:`torch.distributed._shard.sharding_spec.ShardingSpec` object of sharding
            spec for one sharded tensor.
    """
    placements = []
    chunk_sharding_dim = None
    chunk_offset_list = []
    shard_size_list = []
    for shard_metadata in shards_metadata:
        placements.append(shard_metadata.placement)
        local_offsets = shard_metadata.shard_offsets
        chunk_offset_list.append(sum(local_offsets))
        shard_size_list.append(shard_metadata.shard_sizes)
        shard_dims = [idx for idx, e in enumerate(local_offsets) if e != 0]
        if len(shard_dims) == 0:
            continue
        if len(shard_dims) != 1:
            chunk_sharding_dim = None
            break
        if not chunk_sharding_dim:
            chunk_sharding_dim = shard_dims[0]
        elif chunk_sharding_dim != shard_dims[0]:
            chunk_sharding_dim = None
            break
    if chunk_sharding_dim is not None:
        placements = [x for _, x in sorted(zip(chunk_offset_list, placements), key=lambda e: e[0])]
        from .chunk_sharding_spec import ChunkShardingSpec
        chunk_spec = ChunkShardingSpec(dim=chunk_sharding_dim, placements=placements)
        shard_sizes = sorted([x[chunk_sharding_dim] for x in shard_size_list])
        shard_total_length = sum(shard_sizes)
        chunks = len(placements)
        split_size = get_split_size(shard_total_length, chunks)
        chunk_shard_sizes = sorted([get_chunked_dim_size(shard_total_length, split_size, idx) for idx in range(len(placements))])
        if shard_sizes == chunk_shard_sizes:
            return chunk_spec
    return EnumerableShardingSpec(shards_metadata)