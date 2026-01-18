from dataclasses import dataclass
from enum import auto, Enum
from typing import Optional, Sequence, Type
import torch
from torch.nn.modules.batchnorm import _BatchNorm
class ShardingStrategy(Enum):
    """
    This specifies the sharding strategy to be used for distributed training by
    :class:`FullyShardedDataParallel`.

    - ``FULL_SHARD``: Parameters, gradients, and optimizer states are sharded.
      For the parameters, this strategy unshards (via all-gather) before the
      forward, reshards after the forward, unshards before the backward
      computation, and reshards after the backward computation. For gradients,
      it synchronizes and shards them (via reduce-scatter) after the backward
      computation. The sharded optimizer states are updated locally per rank.
    - ``SHARD_GRAD_OP``: Gradients and optimizer states are sharded during
      computation, and additionally, parameters are sharded outside
      computation. For the parameters, this strategy unshards before the
      forward, does not reshard them after the forward, and only reshards them
      after the backward computation. The sharded optimizer states are updated
      locally per rank. Inside ``no_sync()``, the parameters are not resharded
      after the backward computation.
    - ``NO_SHARD``: Parameters, gradients, and optimizer states are not sharded
      but instead replicated across ranks similar to PyTorch's
      :class:`DistributedDataParallel` API. For gradients, this strategy
      synchronizes them (via all-reduce) after the backward computation. The
      unsharded optimizer states are updated locally per rank.
    - ``HYBRID_SHARD``: Apply ``FULL_SHARD`` within a node, and replicate parameters across
      nodes. This results in reduced communication volume as expensive all-gathers and
      reduce-scatters are only done within a node, which can be more performant for medium
      -sized models.
    - ``_HYBRID_SHARD_ZERO2``: Apply ``SHARD_GRAD_OP`` within a node, and replicate parameters across
      nodes. This is like ``HYBRID_SHARD``, except this may provide even higher throughput
      since the unsharded parameters are not freed after the forward pass, saving the
      all-gathers in the pre-backward.
    """
    FULL_SHARD = auto()
    SHARD_GRAD_OP = auto()
    NO_SHARD = auto()
    HYBRID_SHARD = auto()
    _HYBRID_SHARD_ZERO2 = auto()