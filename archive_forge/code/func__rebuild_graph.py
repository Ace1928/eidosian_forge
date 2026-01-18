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
def _rebuild_graph(gm: fx.GraphModule, node_replacements: Dict[torch.fx.Node, torch.fx.GraphModule]) -> None:
    for node in gm.graph.nodes:
        if node not in node_replacements:
            continue
        traced_dispatch = node_replacements[node]
        flatten_args = pytree.arg_tree_leaves(*node.args)
        i, value_remap = (0, {})
        for dtn in traced_dispatch.graph.nodes:
            if dtn.op == OP.PLACEHOLDER:
                value_remap[dtn] = flatten_args[i]
                i += 1
        with gm.graph.inserting_before(node):
            for dtn in traced_dispatch.graph.nodes:
                if dtn.op == OP.PLACEHOLDER:
                    pass
                elif dtn.op == OP.OUTPUT:
                    assert len(dtn.args) == 1, f'Expecting single output, but got {dtn.args} {len(dtn.args[0])}'
                    outputs = dtn.args[0]
                    if len(outputs) == 1:
                        output = outputs[0]
                    else:
                        source = None
                        for i, out in enumerate(outputs):
                            if out is None:
                                continue
                            assert out.op == 'call_function'
                            assert out.target.__module__ == '_operator'
                            assert out.target.__name__ == 'getitem'
                            assert source is None or source == out.args[0]
                            source = out.args[0]
                            assert out.args[1] == i
                        assert source is not None
                        output = source
                    new_node = value_remap[output]
                    node.replace_all_uses_with(new_node)
                else:
                    value_remap[dtn] = gm.graph.node_copy(dtn, lambda n: value_remap[n])
                    if all((isinstance(n.target, torch._ops.OpOverload) and n.target._schema.name.startswith(('aten::_foreach', 'aten::_fused_adam')) for n in [dtn, node])):
                        node.replace_all_uses_with(value_remap[dtn])
                        break
            gm.graph.erase_node(node)
    gm.graph.eliminate_dead_code()
    gm.recompile()