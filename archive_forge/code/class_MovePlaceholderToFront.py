from __future__ import annotations
from typing import List, Optional, Tuple
import torch
import torch.fx
from torch.onnx._internal import _beartype
from torch.onnx._internal.fx import _pass
class MovePlaceholderToFront(_pass.Transform):
    """This pass move all placeholder nodes to the front of the graph node list.

    In torch.fx.Graph, placeholder is a special assignment node. If it's not
    executed in the beginning, it could overwrite values computed by upstream
    nodes.
    """

    @_beartype.beartype
    def _run(self, *args, **kwargs) -> torch.fx.GraphModule:
        graph_module = self.module
        graph = graph_module.graph
        placeholders = []
        first_not_placeholder = None
        for node in graph.nodes:
            if node.op == 'placeholder':
                placeholders.append(node)
            if first_not_placeholder is None and node.op != 'placeholder':
                first_not_placeholder = node
        if first_not_placeholder is None:
            return graph_module
        for placeholder in placeholders:
            first_not_placeholder.prepend(placeholder)
        return graph_module