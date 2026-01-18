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
def _copy_over_q_dq_args(original_node: Node, replacement_node: Node):
    """
    Given a pair of quantize or dequantize nodes, copy over all literal args
    from the original node to the replacement node.
    """
    assert original_node.target == replacement_node.target
    if original_node.target in (torch.ops.quantized_decomposed.quantize_per_tensor.default, torch.ops.quantized_decomposed.dequantize_per_tensor.default):
        start_copy_arg_index = 1
    elif original_node.target in (torch.ops.quantized_decomposed.quantize_per_channel.default, torch.ops.quantized_decomposed.dequantize_per_channel.default):
        start_copy_arg_index = 3
    else:
        raise ValueError("Expected quantize/dequantize nodes, got '%s'" % original_node.target)
    replacement_node.args = replacement_node.args[:start_copy_arg_index] + original_node.args[start_copy_arg_index:]