import operator
import types
import torch
from torch._export import capture_pre_autograd_graph
from torch.fx import (
from torch.nn.utils.fusion import fuse_conv_bn_weights
from typing import Any, Callable, Dict, Optional, Tuple, List, Union
from torch.utils._pytree import LeafSpec
from torch.ao.quantization.fx._decomposed import quantized_decomposed_lib  # noqa: F401
from torch.ao.quantization.quantizer import QuantizationAnnotation
def _is_sym_size_node(node: Node):
    return node.op == 'call_function' and node.target == torch.ops.aten.sym_size.default or node.target == torch.ops.aten.sym_numel.default or node.target == torch.ops.aten.sym_numel or (node.target == torch.ops.aten.sym_size)