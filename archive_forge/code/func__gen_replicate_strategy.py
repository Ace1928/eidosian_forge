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
def _gen_replicate_strategy(mesh: DeviceMesh, input_specs: Optional[List[DTensorSpec]]=None) -> PlacementStrategy:
    """Util function to generate a replicate strategy."""
    return PlacementStrategy(output_spec=DTensorSpec(mesh=mesh, placements=(Replicate(),)), input_specs=input_specs)