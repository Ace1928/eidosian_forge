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
def _get_quantized_conv_bn_example_inputs_kwargs(is_per_channel: bool, has_bias: bool, is_cuda: bool) -> Dict[str, Any]:
    """
    Optional example inputs for quantized and folded conv-bn patterns
    used in convert, expressed as kwargs.
    """
    kwargs = {}
    if is_per_channel:
        kwargs['scale'] = torch.tensor([1], dtype=torch.float)
        kwargs['zero_point'] = torch.tensor([0], dtype=torch.int)
    if has_bias:
        kwargs['conv_bias'] = torch.randn(1)
    if is_cuda:
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                kwargs[k] = v.cuda()
    return kwargs