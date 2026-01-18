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
class DataParallelStrategy(OpStrategy):
    """DataParallelStrategy is a special case of OpStrategy that only records the "data parallel style" placement
    strategy for each fx Node.

    It takes a list of PlacementStrategy, where each PlacementStrategy describes
    one way to distribute the tensor and computation. In the DataParallel case,
    there're two possible ways to distribute the parameters:
        1. replicate the parameter over a set of devices (DDP like behavior)
        2. shard the parameter on its tensor dimension 0 over a set of devices
           (FSDP like behavior).

    In addition to the strategy list, we also need to:
    1. `node_type`: record the type of each node in the graph, so that we can
        determine how to propagate in a data parallel fashion.
    2. `reduce_over_batch` is specifically tied to data parallel as the loss
        calculation usually results in scalar tensor where it comes from a
        reduction over the batch dimension. We need to know this information
        so that we could keep the output as sharded.
    """

    def __init__(self, node_type: NodeType, strategy_list: List[PlacementStrategy], reduction_over_batch: bool=False):
        super().__init__(strategy_list)
        self.node_type = node_type
        self.reduction_over_batch = reduction_over_batch

    def __str__(self) -> str:
        return f'type: {self.node_type}, {super().__str__()}'