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
def _is_supported_batch_norm_for_training(node: Node):
    """
    Return True if the given node refers to an aten batch norm op QAT supports.
    """
    supported_ops = [torch.ops.aten._native_batch_norm_legit.default, torch.ops.aten.cudnn_batch_norm.default, torch.ops.aten.miopen_batch_norm.default]
    return node.target in supported_ops