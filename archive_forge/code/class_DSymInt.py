import logging
import operator
from dataclasses import dataclass
from enum import auto, Enum
from functools import partial
from typing import Any, Callable, cast, Dict, List, Optional, Sequence, Tuple, Union
import torch
import torch.distributed._spmd.experimental_ops
import torch.fx as fx
from torch.distributed._spmd.comm_tensor import _get_tracer
from torch.distributed._spmd.graph_utils import OP
from torch.distributed._spmd.log_utils import get_logger
from torch.distributed._tensor import DeviceMesh, DTensor
from torch.distributed._tensor.op_schema import OpSchema
from torch.distributed._tensor.placement_types import (
from torch.distributed._tensor.redistribute import redistribute_local_tensor
from torch.fx.experimental.proxy_tensor import make_fx, proxy_slot
from torch.utils import _pytree as pytree
from torch.utils._pytree import tree_flatten, tree_map, tree_map_only, tree_unflatten
@dataclass
class DSymInt:
    """DSymInt represents a value retrieved by a SymInt op from a DTensor.

    DSymInt helps View and Factory ops to determine the placement and shape of the
    output tensor, as those operators either do not have an input DTensor or
    the input DTensor is insufficient to determine the output tensor's placement.
    """
    global_value: int
    local_value: int
    mesh: DeviceMesh

    def is_shard(self) -> bool:
        return self.local_value != self.global_value

    @classmethod
    def from_node(cls, node: fx.Node, dtensor: DTensor) -> 'DSymInt':
        dim: int = 0
        if node.target == aten.sym_size:
            dim = cast(int, node.args[1])
            return cls(global_value=dtensor.size(dim), local_value=dtensor.to_local().size(dim), mesh=dtensor.device_mesh)
        elif node.target == aten.sym_numel:
            return cls(global_value=dtensor.numel(), local_value=dtensor.to_local().numel(), mesh=dtensor.device_mesh)
        elif node.target == aten.sym_stride:
            dim = cast(int, node.args[1])
            return cls(global_value=dtensor.stride(dim), local_value=dtensor.to_local().stride(dim), mesh=dtensor.device_mesh)
        else:
            raise NotImplementedError(f'DSymInt does not support {node.target}')