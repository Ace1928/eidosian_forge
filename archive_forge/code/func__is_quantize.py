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
def _is_quantize(n: Node) -> bool:
    return n.target in [torch.ops.quantized_decomposed.quantize_per_tensor.default, torch.ops.quantized_decomposed.quantize_per_tensor.tensor, torch.ops.quantized_decomposed.quantize_per_channel.default]