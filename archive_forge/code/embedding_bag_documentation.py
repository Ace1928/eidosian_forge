from typing import cast, List
import torch
import torch.distributed as dist
from torch._C._distributed_c10d import ReduceOp
from torch.distributed._shard.sharded_tensor import ShardedTensor
from torch.distributed._shard.sharding_spec import ChunkShardingSpec
from torch.distributed._shard.sharding_spec.api import custom_sharding_spec_op
from torch.distributed.nn.functional import all_gather, reduce_scatter
from ._common import (

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
    