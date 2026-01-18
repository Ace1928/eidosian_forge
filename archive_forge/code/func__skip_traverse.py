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
def _skip_traverse(self, all_nodes: NodeList, skip_nodes: List) -> NodeSet:
    """
        Skip certain nodes in graph based on settings
        """
    start_idx = 0
    num_nodes = len(all_nodes)
    idx = 0
    culprits = set()
    while idx < num_nodes:
        node = all_nodes[idx]
        if node.name in skip_nodes:
            if idx > start_idx:
                culprits = self._skip_traverse_impl(all_nodes, start_idx, idx)
            start_idx = idx + 1
        elif idx == num_nodes - 1 and start_idx <= idx:
            culprits = self._skip_traverse_impl(all_nodes, start_idx, idx + 1)
        idx += 1
    return culprits