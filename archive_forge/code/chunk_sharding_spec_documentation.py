from dataclasses import dataclass
import torch
import torch.distributed._shard.sharded_tensor.metadata as sharded_tensor_meta
from torch.distributed._shard.metadata import ShardMetadata
from torch.distributed._shard.sharded_tensor.shard import Shard
from torch.distributed._shard.sharded_tensor.utils import (
from torch.distributed._shard._utils import narrow_tensor
import torch.distributed as dist
import torch.distributed.distributed_c10d as distributed_c10d
from typing import List, Union, TYPE_CHECKING
from ._internals import (
from .api import ShardingSpec

        Args:
            src_rank: group rank relative to ``process_group``

            N.B. If ``process_group`` is None, ``src_rank`` is a global rank.
        