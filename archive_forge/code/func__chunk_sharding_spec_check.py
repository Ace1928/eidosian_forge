import torch
import torch.distributed as dist
from torch.distributed._shard.sharded_tensor import ShardedTensor
from torch.distributed._shard.sharded_tensor._ops._common import _sharded_op_common
from torch.distributed._shard.sharding_spec import ChunkShardingSpec
from torch.distributed._shard.sharding_spec._internals import (
from torch.distributed._shard.sharding_spec.api import custom_sharding_spec_op
from torch.distributed.nn.functional import (
def _chunk_sharding_spec_check(spec, op):
    """
    For the given op implementation check if the sharding spec is ChunkShardingSpec.
    """
    if not isinstance(spec, ChunkShardingSpec):
        raise NotImplementedError(f"Only ChunkShardingSpec supported for '{op.__name__}'.")