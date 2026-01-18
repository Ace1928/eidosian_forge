import logging
import torch
from torch._export.pass_base import _ExportPassBase
from torch.ao.quantization.pt2e.utils import (
from torch.fx.node import map_arg
from torch.fx.passes.infra.pass_base import PassResult
def _maybe_duplicate_dq(gm: torch.fx.GraphModule, dq_node: torch.fx.Node, user: torch.fx.Node):
    annotation = user.meta.get('quantization_annotation', None)
    if not _is_valid_annotation(annotation):
        return
    with gm.graph.inserting_after(dq_node):
        new_node = gm.graph.node_copy(dq_node)

        def maybe_replace_node(n: torch.fx.Node) -> torch.fx.Node:
            if n == dq_node:
                return new_node
            else:
                return n
        new_args = map_arg(user.args, maybe_replace_node)
        new_kwargs = map_arg(user.kwargs, maybe_replace_node)
        user.args = new_args
        user.kwargs = new_kwargs