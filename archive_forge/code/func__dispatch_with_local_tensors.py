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
def _dispatch_with_local_tensors(op: torch._ops.OpOverload, local_args: Tuple[Any, ...], kwargs: Optional[Dict[str, Any]]=None, specs: Optional[Dict[torch.Tensor, Tuple[torch.Size, DeviceMesh, Sequence[Placement], Sequence[Placement]]]]=None) -> Any:
    if kwargs is None:
        kwargs = {}
    if specs is None:
        specs = {}

    def redistribute(arg: Any) -> Any:
        tensor_shape, mesh, current_placement, target_placement = specs[arg]
        tensor_meta = TensorMeta(tensor_shape, stride=arg.stride(), dtype=arg.dtype)
        current_spec = DTensorSpec(mesh, tuple(current_placement), tensor_meta=tensor_meta)
        target_spec = DTensorSpec(mesh, tuple(target_placement), tensor_meta=tensor_meta)
        return redistribute_local_tensor(arg, current_spec, target_spec) if isinstance(arg, torch.Tensor) and arg in specs else arg
    return op(*tree_map(redistribute, local_args), **kwargs)