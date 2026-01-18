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
def _update_specs_for_redistribute(args, target_schema, redistribute):
    flatten_args, args_tree_spec = tree_flatten(args)
    flatten_args_schema = pytree.tree_leaves(target_schema.args_schema)
    specs: Dict[torch.Tensor, Tuple[torch.Size, DeviceMesh, Sequence[Placement], Sequence[Placement]]] = {}
    for i, arg in enumerate(flatten_args):
        if isinstance(arg, DTensor):
            if redistribute:
                specs[arg._local_tensor] = (arg.size(), flatten_args_schema[i].mesh, arg.placements, flatten_args_schema[i].placements)
            flatten_args_schema[i] = arg._local_tensor
    unflattened_args = tree_unflatten(flatten_args_schema, args_tree_spec)
    return (specs, unflattened_args)