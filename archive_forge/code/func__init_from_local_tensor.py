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
def _init_from_local_tensor(cls, local_tensor: torch.Tensor, sharding_spec: shard_spec.ShardingSpec, *global_size: Sequence[int], process_group: Optional[dist.ProcessGroup]=None, init_rrefs=False) -> ShardedTensor:
    """
        Initialize a ShardedTensor given only one local tensor, global sharded tensor
        size and sharding spec on each rank.

        Args:
            local_tensor (Tensor): Single tensor of local shard stored in each rank.
            sharding_spec (:class:`torch.distributed._shard.sharding_spec.ShardingSpec`):
                The specification describing how to shard the Tensor.
            global_size (Sequence[int]): Size of the sharded tensor.
            process_group (ProcessGroup, optional): The process group to aggregate on.
                Default: None
            init_rrefs (bool, optional): Whether or not to initialize
                :class:`torch.distributed.rpc.RRef`s pointing to remote shards.
                Need to initialize the RPC Framework if specified as ``True``.
                Default: ``False``.

        Returns:
            A :class:`ShardedTensor` sharded based on the given sharding_spec with local
                tensor stored in the current rank.

        Examples:
            >>> # xdoctest: +SKIP
            >>> # All tensors below are of torch.int64 type.
            >>> # We have 2 process groups, 2 ranks.
            >>> tensor = torch.arange(2, dtype=torch.int64) + 1 + 2 * rank
            >>> local_tensor = torch.unsqueeze(torch.cat([tensor, tensor + 2]))
            >>> local_tensor
            tensor([[1, 2, 3, 4]]) # Rank 0
            tensor([[3, 4, 5, 6]]) # Rank 1
            >>> sharding_dim = 0
            >>> sharding_spec = ChunkShardingSpec(
                    dim=sharding_dim,
                    placements=[
                        "rank:0/cuda:0",
                        "rank:1/cuda:1",
                    ],
                )
            >>> st = ShardedTensor._init_from_local_tensor(local_tensor, sharding_spec, [2, 4])
            >>> st
            ShardedTensor(
                ShardedTensorMetadata(
                    shards_metadata=[
                        ShardMetadata(shard_offsets=[0, 0], shard_sizes=[1, 4], placement=rank:0/cuda:0),
                        ShardMetadata(shard_offsets=[1, 0], shard_sizes=[1, 4], placement=rank:1/cuda:1),
                    ],
                    size=torch.Size([2, 4])
            )
            >>> st.local_tensor()
            tensor([1, 2, 3, 4]) # Rank 0
            tensor([3, 4, 5, 6]) # Rank 1

        Warning: This API is experimental and subject to change. It lacks of a fully across
                 rank validations, and we only validate the local shard on the current rank.
                 We fully rely on the user to ensure local tensor is sharded based on the
                 sharding spec.
        """
    warnings.warn(DEPRECATE_MSG)
    if not local_tensor.is_contiguous():
        raise ValueError('local_tensor is not a contiguous Tensor.')
    global_tensor_size = _flatten_tensor_size(global_size)
    tensor_properties = TensorProperties(dtype=local_tensor.dtype, layout=local_tensor.layout, requires_grad=local_tensor.requires_grad, memory_format=torch.contiguous_format, pin_memory=local_tensor.is_pinned())
    sharded_tensor_metadata = sharding_spec.build_metadata(global_tensor_size, tensor_properties)
    process_group = process_group if process_group is not None else distributed_c10d._get_default_group()
    current_rank = dist.get_rank(process_group)
    local_shards: List[Shard] = []
    for shard_metadata in sharded_tensor_metadata.shards_metadata:
        rank, device = _parse_and_validate_remote_device(process_group, shard_metadata.placement)
        if rank == current_rank:
            local_shards.append(Shard(local_tensor, shard_metadata))
    return ShardedTensor._init_from_local_shards_and_global_metadata(local_shards, sharded_tensor_metadata, process_group=process_group, init_rrefs=init_rrefs, sharding_spec=sharding_spec)