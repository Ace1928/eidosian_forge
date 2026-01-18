import operator
from contextlib import contextmanager
from enum import Enum
from typing import Any, cast, Dict, List, Optional, Tuple
import torch
import torch.distributed.distributed_c10d as c10d
import torch.fx as fx
import torch.library
import torch.nn as nn
import torch.utils._pytree as pytree
from torch.distributed._spmd.batch_dim_utils import BatchDimAnalyzer
from torch.distributed._tensor import DeviceMesh, distribute_tensor, Replicate, Shard
from torch.distributed._tensor._utils import compute_local_shape
from torch.distributed._tensor.op_schema import (
from torch.distributed._tensor.placement_types import _Partial, DTensorSpec, Placement
from torch.distributed._tensor.redistribute import redistribute_local_tensor
from torch.fx import GraphModule
from torch.fx.experimental.proxy_tensor import make_fx
from torch.fx.passes.shape_prop import _extract_tensor_metadata
from torch.nn.utils._named_member_accessor import NamedMemberAccessor
class DataParallelStyle(Enum):
    """This enum represents the style of the data-parallel operation.

    We have three types of Data Parallel style:
    1. DEFAULT: the default data parallel style, which is to represent a mixed
                replicate and fully shard behavior. For each parameter that is able
                to be sharded evenly, we shard it, otherwise we would replicate the
                parameter. This style avoids potential padding if the parameters
                cannot be sharded evenly, but it would generate a mixed of all_reduce
                and reduce_scatter.
    2. REPLICATE: the data parallel style that replicates all model parameters.
                  This is similar to the behavior of DistributedDataParallel.
    3. FULLY_SHARD: the data parallel style that shards all model parameters. This
                    is similar to the behavior of FullyShardedDataParallel, the
                    difference is that FullyShardedDataParallel (ZERO-3), which
                    shards the model using FlatParameter based sharding,
                    while this style shards each parameter into DTensor.
    """
    DEFAULT = 0
    REPLICATE = 1
    FULLY_SHARD = 2