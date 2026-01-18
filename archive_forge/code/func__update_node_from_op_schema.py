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
def _update_node_from_op_schema(node: torch.fx.Node, op_schema: OpSchema) -> None:
    flat_args, args_tree_spec = tree_flatten(node.args)
    flat_args_schema = pytree.tree_leaves(op_schema.args_schema)

    def is_sym_int_or_int(arg: Union[int, torch.fx.Node]) -> bool:
        if isinstance(arg, torch.fx.Node):
            return arg.target in [aten.sym_size, aten.sym_numel, aten.sym_stride]
        return isinstance(arg, int)
    assert len(flat_args) == len(flat_args_schema)
    for i, (arg, arg_schema) in enumerate(zip(flat_args, flat_args_schema)):
        if is_sym_int_or_int(arg) and isinstance(arg_schema, int):
            flat_args[i] = arg_schema
    args = tree_unflatten(flat_args, args_tree_spec)
    for idx, arg in enumerate(args):
        node.update_arg(idx, arg)
    return None