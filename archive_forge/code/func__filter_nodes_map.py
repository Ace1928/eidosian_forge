import dataclasses
import itertools
import operator
from typing import Any, Callable, Dict, List, Tuple
import torch
from torch.fx import Graph, GraphModule, Node
from torch.fx.subgraph_rewriter import (
import torch.nn.functional as F
from torch.ao.quantization.fx._decomposed import quantized_decomposed_lib  # noqa: F401
from torch.ao.quantization.quantizer import (
from .utils import (
def _filter_nodes_map(nodes_map: Dict[Node, Node]) -> Dict[Node, Node]:
    """
    Return a filtered `nodes_map` returned from the subgraph rewriter.
    The filtered `nodes_map` will contain only nodes that are actually
    matched in the pattern, excluding None or placeholder nodes.
    """
    new_nodes_map: Dict[Node, Node] = {}
    for pattern_node, graph_node in nodes_map.items():
        if graph_node is None:
            continue
        if pattern_node.op == 'placeholder':
            continue
        new_nodes_map[pattern_node] = graph_node
    return new_nodes_map