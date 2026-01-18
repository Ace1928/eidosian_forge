import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple
import torch
import torch.fx
from torch.fx._compatibility import compatibility
from torch.fx.node import map_arg
from .shape_prop import ShapeProp
from .split_utils import split_by_tags
from .tools_common import (
def _collect_nodes(self, start: Optional[str], end: Optional[str]) -> NodeList:
    """
        Collect nodes in the model that between nodes with name of `start` and `end`.
        These two nodes are also included.
        """
    nodes: NodeList = []
    add_node = start is None
    for node in self.module.graph.nodes:
        if node.op not in CALLABLE_NODE_OPS:
            continue
        if node.name == start:
            add_node = True
        if add_node:
            nodes.append(node)
        if node.name == end:
            break
    return nodes