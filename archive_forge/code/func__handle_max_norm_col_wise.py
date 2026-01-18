import torch
import torch.distributed as dist
from torch.distributed._shard.sharded_tensor import ShardedTensor
from torch.distributed._shard.sharded_tensor._ops._common import _sharded_op_common
from torch.distributed._shard.sharding_spec import ChunkShardingSpec
from torch.distributed._shard.sharding_spec._internals import (
from torch.distributed._shard.sharding_spec.api import custom_sharding_spec_op
from torch.distributed.nn.functional import (
def _handle_max_norm_col_wise(max_norm, norm_type, local_shard, input, world_size, gathered_inputs, pg):
    """
    For col-wise sharding of weight, we need to aggregate the
    norm across all ranks before we can perform the proper re-norm.
    Note that, the max_norm logic is only applied to the embedding
    indices that are looked up and not the whole shard.

    Args:
        max_norm: If given, each embedding vector with norm larger
            than max_norm is renormalized to have norm max_norm.
            Note: this will modify weight in-place.
        norm_type: The p in the p-norm to compute for the max_norm option.
        local_shard: col-wise shared local weight used for lookup.
        input: tensor to be applied op to.
        world_size: number of ranks.
        gathered_inputs: list of inputs from all ranks.
        pg: process group.

    Return:
        local_shard_norm_renormed: local_shard re-normed to max_norm if the norm is larger
            than it.

    """
    norm_type = norm_type if norm_type is not None else 2.0
    unique_inp = torch.unique(torch.cat(gathered_inputs))
    local_shard_sum = torch.sum(torch.pow(torch.abs(local_shard), norm_type), dim=1, dtype=local_shard.dtype)
    local_shard_sum = all_reduce(local_shard_sum, group=pg)
    local_shard_norm = torch.pow(local_shard_sum, 1.0 / norm_type)
    max_norm_tensor = torch.full((local_shard.size(0),), float('inf'), dtype=local_shard.dtype, device=input.device)
    max_norm_tensor[unique_inp] = max_norm
    local_shard_t = local_shard.t().contiguous()
    normalized_tensor = torch.where(local_shard_norm > max_norm_tensor, max_norm_tensor, local_shard_norm)
    local_shard_norm[local_shard_norm == 0.0] = 1.0
    local_shard_norm_renormed = torch.div(torch.mul(local_shard_t, normalized_tensor), local_shard_norm).t().contiguous()
    return local_shard_norm_renormed