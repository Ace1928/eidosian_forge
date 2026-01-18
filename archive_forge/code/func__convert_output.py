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
def _convert_output(gm: fx.GraphModule, node: fx.Node, node_to_obj: Dict[fx.Node, Any]) -> fx.Node:
    new_args = []
    has_partial = False
    for argument in node.args[0]:
        if not isinstance(argument, fx.Node):
            new_args.append(argument)
            continue
        obj = node_to_obj[argument]
        if not _is_partial_dtensor(obj):
            new_args.append(argument)
            continue
        has_partial = True
        dt = cast(DTensor, obj)
        traced_dispatch, result_obj = _build_dummy_add_graph(dt, node_to_obj)
        wait = [n for n in traced_dispatch.graph.nodes if n.name == 'wait_comm' or n.name == 'wait_tensor']
        add = [n for n in traced_dispatch.graph.nodes if n.name == 'add']
        assert len(wait) == 1 and len(add) == 1
        add[0].replace_all_uses_with(wait[0])
        traced_dispatch.graph.eliminate_dead_code()
        node_to_obj[wait[0]] = result_obj
        value_remap: Dict[fx.Node, fx.Node] = {}
        for dtn in traced_dispatch.graph.nodes:
            if dtn.op == OP.PLACEHOLDER:
                value_remap[dtn] = argument
            elif dtn.op == OP.OUTPUT:
                assert len(dtn.args) == 1 and len(dtn.args[0]) == 1, f'Expecting single output, but got {dtn.args} {len(dtn.args)}'
                new_args.append(value_remap[dtn.args[0][0]])
                node_to_obj[value_remap[dtn.args[0][0]]] = node_to_obj[dtn.args[0][0]]
            else:
                if dtn.op == OP.GET_ATTR:
                    setattr(gm, dtn.target, getattr(traced_dispatch, dtn.target))
                with gm.graph.inserting_before(node):
                    value_remap[dtn] = gm.graph.node_copy(dtn, lambda n: value_remap[n])
    if has_partial:
        gm.graph.erase_node(node)
        return gm.graph.output(new_args)
    else:
        return node