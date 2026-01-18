from typing import Callable, Dict, List, Set
import torch
import torch.fx as fx
import torch.utils._pytree as pytree
from torch import Tensor
from torch.distributed._tensor import DeviceMesh, Replicate, Shard
from torch.distributed._tensor.ops.view_ops import (
from torch.distributed._tensor.placement_types import _Partial, DTensorSpec
Compute the batch dimension for the current node, then generate the sharding spec that shards on the batch dimension.