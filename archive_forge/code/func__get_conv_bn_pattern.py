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
def _get_conv_bn_pattern(conv_fn: Callable) -> Callable:

    def _conv_bn_pattern(x: torch.Tensor, conv_weight: torch.Tensor, conv_bias: torch.Tensor, bn_weight: torch.Tensor, bn_bias: torch.Tensor, bn_running_mean: torch.Tensor, bn_running_var: torch.Tensor) -> torch.Tensor:
        x = conv_fn(x, conv_weight, conv_bias)
        x = F.batch_norm(x, bn_running_mean, bn_running_var, bn_weight, bn_bias, training=True)
        return x
    return _conv_bn_pattern