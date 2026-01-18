import torch
import torch.distributed as dist
from torch.distributed._shard.sharded_tensor import ShardedTensor
from torch.distributed._shard.sharding_spec import ChunkShardingSpec
from torch.distributed._shard.sharding_spec.api import custom_sharding_spec_op
from torch.distributed.nn.functional import all_gather, reduce_scatter
from ._common import (
def _handle_row_wise_sharding(input, world_size, weight, local_shard, max_norm, norm_type, padding_idx, rank, pg):
    """
    Entry-point function to handle the logic of row-wise sharding of weight
    for embedding. (Detailed explanations of the logic can be found in
    the comment for sharded_embedding.)

    Args:
        input: list of ID used for lookup and aggregation.
        world_size: number of ranks.
        weight: sharded weight tensor.
        local_shard: row-wise shared local weight used for lookup.
        max_norm: If given, each embedding vector with norm larger
            than max_norm is renormalized to have norm max_norm.
            Note: this will modify weight in-place.
        norm_type: The p in the p-norm to compute for the max_norm option.
        padding_idx: If specified, the entries at padding_idx do
            not contribute to the gradient; therefore, the embedding
            vector at padding_idx is not updated during training,
            i.e. it remains as a fixed “pad”.
        rank: # of cuda process.
        pg: process group.

    Returns: final result of lookup.
    """
    gather_inp = _all_gather_base_input(input, pg)
    lookup_input, padding_idx, padding_row = _handle_row_wise_mask(gather_inp, padding_idx, weight, world_size, rank)
    if max_norm is not None:
        torch.nn.functional.embedding(torch.unique(lookup_input)[:-1], local_shard, padding_idx=padding_idx, max_norm=max_norm, norm_type=norm_type)
        max_norm = None
    local_input_embeddings = torch.nn.functional.embedding(lookup_input, torch.cat([local_shard, padding_row]), padding_idx=padding_idx, max_norm=max_norm, norm_type=norm_type)
    local_shards = local_input_embeddings.chunk(pg.size())
    return reduce_scatter(torch.empty_like(local_shards[0]), list(local_shards), group=pg)