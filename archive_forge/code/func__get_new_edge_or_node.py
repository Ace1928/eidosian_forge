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
def _get_new_edge_or_node(edge_or_node: EdgeOrNode):
    if isinstance(edge_or_node, Node):
        _node = edge_or_node
        return original_to_replacement_node.get(_node, _node)
    elif isinstance(edge_or_node, tuple) and len(edge_or_node) == 2 and all((isinstance(x, Node) for x in edge_or_node)):
        src, dest = edge_or_node
        return (original_to_replacement_node.get(src, src), original_to_replacement_node.get(dest, dest))
    else:
        raise ValueError('unexpected type for edge_or_node: ', type(edge_or_node))