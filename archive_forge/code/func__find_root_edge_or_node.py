import torch
from torch._subclasses import FakeTensor
from torch.ao.quantization.fx.prepare import (
from torch.fx import (
from torch.fx.node import Argument
from torch.ao.quantization import QConfigMapping
from torch.ao.quantization.qconfig import QConfigAny
from torch.ao.quantization.fx.custom_config import PrepareCustomConfig
from typing import Dict, Tuple, Union, Any, Optional
from torch.ao.quantization.quantizer import (
from torch.ao.quantization import ObserverOrFakeQuantize
def _find_root_edge_or_node(edge_or_node: EdgeOrNode, shared_with_map: Dict[EdgeOrNode, EdgeOrNode]) -> EdgeOrNode:
    """Find the root node for the sharing tree
    Args:
        edge_or_node: edge/node that we want to find the root
        shared_with_map: each edge/node points to the parent, the root node will points to itself

    Returns:
        root edge/node
    """
    parent = shared_with_map[edge_or_node]
    if parent == edge_or_node:
        return edge_or_node
    root = _find_root_edge_or_node(parent, shared_with_map)
    shared_with_map[edge_or_node] = root
    return root