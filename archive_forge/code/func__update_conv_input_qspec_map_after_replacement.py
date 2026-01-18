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
def _update_conv_input_qspec_map_after_replacement(original_node: Node, replacement_node: Node):
    """
    Update the `input_qspec_map` in the annotation after subgraph rewriting.

    The original annotation referred to the nodes in the original graph,
    so the keys in the `input_qspec_map` will need to be updated to reflect
    the corresponding nodes in the replacement graph.
    """
    assert _is_conv(original_node)
    assert _is_conv(replacement_node)
    if 'quantization_annotation' not in original_node.meta:
        return
    original_input_qspec_map = original_node.meta['quantization_annotation'].input_qspec_map
    input_qspec_map = {}
    all_configs = list(original_input_qspec_map.items())
    input_qspec_map[replacement_node.args[0]] = all_configs[0][1]
    input_qspec_map[replacement_node.args[1]] = all_configs[1][1]
    if len(replacement_node.args) > 2 and len(all_configs) > 2:
        input_qspec_map[replacement_node.args[2]] = all_configs[2][1]
    replacement_node.meta['quantization_annotation'].input_qspec_map = input_qspec_map