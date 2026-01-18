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
def _no_conv_bias_filter(match: 'InternalMatch', original_graph: Graph, pattern_graph: Graph) -> bool:
    """
    Match filter for the subgraph rewriter that returns True if the conv node in
    the original graph does NOT have bias.
    """
    return not _has_conv_bias_filter(match, original_graph, pattern_graph)