from typing import cast, List
import torch
import torch.distributed as dist
from torch._C._distributed_c10d import ReduceOp
from torch.distributed._shard.sharded_tensor import ShardedTensor
from torch.distributed._shard.sharding_spec import ChunkShardingSpec
from torch.distributed._shard.sharding_spec.api import custom_sharding_spec_op
from torch.distributed.nn.functional import all_gather, reduce_scatter
from ._common import (
def _validate_embedding_bag_param(args, kwargs):
    """
    Validate input params of sharded embeddingBag op.

    Args:
        input: list of ID used for lookup and aggregation.
        weight: sharded weight tensor.
        kwargs: same as normal EmbeddingBag.

    Return: None.
    """
    input = args[0]
    weight = args[1]
    offsets = kwargs.get('offsets')
    per_sample_weights = kwargs.get('per_sample_weights')
    mode = kwargs.get('mode')
    max_norm = kwargs.get('max_norm')
    scale_grad_by_freq = kwargs.get('scale_grad_by_freq')
    sparse = kwargs.get('sparse')
    include_last_offset = kwargs.get('include_last_offset')
    if not isinstance(input, torch.Tensor):
        raise TypeError('input need to be torch.Tensor')
    if offsets is not None and (not isinstance(offsets, torch.Tensor)):
        raise TypeError('offsets need to be torch.Tensor')
    if per_sample_weights is not None and (not isinstance(per_sample_weights, torch.Tensor)):
        raise TypeError('per_sample_weights need to be torch.Tensor')
    if not isinstance(weight, ShardedTensor):
        raise TypeError('weight needs to be ShardedTensor')
    if len(input.size()) > 2:
        raise ValueError('Input more than 2 dims not supported')
    weight_size = weight.size()
    if len(weight_size) != 2:
        raise ValueError('Weight needs to have exactly 2 dims')
    if int(torch.min(input).item()) < 0:
        raise ValueError('Index out of range in Input %d %d', int(torch.min(input).item()), weight_size[1])
    if int(torch.max(input).item()) >= weight_size[0]:
        raise ValueError('Index out of range in Input %d %d', int(torch.max(input).item()), weight_size[1])
    if offsets is not None and len(input.size()) != 1:
        raise ValueError('Input dimension needs to be exactly 1 dim')
    if len(input.size()) == 1 and offsets is None:
        raise ValueError('offsets is required for 1D input')
    if per_sample_weights is not None and per_sample_weights.size() != input.size():
        raise ValueError(f'per_sample_weights size {per_sample_weights.size()} not equal to input size {input.size()}')
    if mode is None:
        mode = 'mean'
    if mode not in ['sum', 'mean', 'max']:
        raise ValueError(f"mode '{mode}' is not supported")
    if scale_grad_by_freq:
        raise RuntimeError('nn.Embedding weight sharded with flag on "scale_grad_by_freq" not supported!')
    if sparse:
        raise RuntimeError('nn.Embedding weight sharded with flag on "sparse" not supported!')
    if include_last_offset and offsets is None:
        raise ValueError('offsets is required for flag "include_last_offset"!')
    if include_last_offset and cast(List[int], offsets)[-1] != input.size(0):
        raise ValueError('offsets need to have the input size in the end when the flag "include_last_offset" is on!')
    if max_norm and max_norm <= 0.0:
        raise ValueError('"max_norm" must be larger than zero!')
    if not isinstance(weight._sharding_spec, ChunkShardingSpec):
        raise ValueError('Only ChunkShardingSpec supported for ShardedTensor ops!')
    if len(weight.local_shards()) != 1:
        raise ValueError('Only one local shard supported!')