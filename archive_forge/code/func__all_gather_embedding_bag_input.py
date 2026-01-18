from typing import cast, List
import torch
import torch.distributed as dist
from torch._C._distributed_c10d import ReduceOp
from torch.distributed._shard.sharded_tensor import ShardedTensor
from torch.distributed._shard.sharding_spec import ChunkShardingSpec
from torch.distributed._shard.sharding_spec.api import custom_sharding_spec_op
from torch.distributed.nn.functional import all_gather, reduce_scatter
from ._common import (
def _all_gather_embedding_bag_input(input, per_sample_weights, offsets, pg):
    """
    In case we need to gather input and all other parameters of embeddingBag
    ops, we need to stack all input together to perform ``all_gather``
    collective communication just once.

    Note that since offsets does not share the same size as input and
    is always smaller than input, we resize it during the communication.

    Args:
        input: tensor to be applied op on.
        per_sample_weights: weights for weighted sum mode.
        offsets: when input is 1D. offsets determines the starting
            index position of each bag (sequence) in input.
        pg: process group.

    Returns:
        gathered_inputs: list of input tensor gathered from each rank.
        gathered_per_sample_weights: list of per_sample_weights from each rank.
        gathered_offsets: list of offsets from each rank.
    """
    input_to_gather = [input]
    if per_sample_weights is not None:
        input_to_gather.append(per_sample_weights)
    if offsets is not None:
        input_to_gather.append(offsets.clone().resize_(input.size()))
    gathered_inputs = all_gather(torch.stack(input_to_gather), group=pg)
    gathered_per_sample_weights = None
    if per_sample_weights is not None:
        gathered_per_sample_weights = [t[1] for t in gathered_inputs]
    gathered_offsets = None
    if offsets is not None:
        idx = 2 if per_sample_weights is not None else 1
        gathered_offsets = [t[idx].resize_(offsets.size()).to(offsets.dtype) for t in gathered_inputs]
    gathered_inputs = [t[0].to(input.dtype) for t in gathered_inputs]
    return (gathered_inputs, gathered_per_sample_weights, gathered_offsets)