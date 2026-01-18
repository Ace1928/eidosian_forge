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
def _fold_conv_bn_qat(m: GraphModule) -> GraphModule:
    m = _fold_conv_bn_qat_helper(m, F.conv1d, _quantized_conv1d_bn_example_inputs, is_cuda=False)
    m = _fold_conv_bn_qat_helper(m, F.conv2d, _quantized_conv2d_bn_example_inputs, is_cuda=False)
    if torch.cuda.is_available():
        m = _fold_conv_bn_qat_helper(m, F.conv1d, _quantized_conv1d_bn_example_inputs, is_cuda=True)
        m = _fold_conv_bn_qat_helper(m, F.conv2d, _quantized_conv2d_bn_example_inputs, is_cuda=True)
    return m