import functools
import operator
from functools import reduce
from typing import Any, Tuple
import torch
from torch.fx.experimental.symbolic_shapes import has_free_symbols
from .. import ir
from ..lowering import lowerings as L
from ..pattern_matcher import (
from ..virtualized import ops
from .freezing_patterns import register_freezing_graph_pattern
from .post_grad import register_lowering_pattern
from .quantization import (
def _is_packable_linear(match):
    """
        Check if the node is supported for MKLDNN linear.
        """
    linear_node = match.output_node()
    weight_idx = 2 if linear_node.target == aten.addmm.default else 1
    if linear_node.args[weight_idx].op != 'get_attr':
        return False
    input_meta_value = linear_node.args[weight_idx - 1].meta.get('val')
    weight_meta_value = linear_node.args[weight_idx].meta.get('val')
    if input_meta_value is None or weight_meta_value is None:
        return False
    batch_size = input_meta_value.shape[0]
    is_bf16_weight = weight_meta_value.dtype == torch.bfloat16
    if not is_bf16_weight and (not torch._C.has_mkl or has_free_symbols(batch_size)):
        return False
    for meta_value in [input_meta_value, weight_meta_value]:
        if meta_value is None or meta_value.device.type != 'cpu' or meta_value.dim() != 2:
            return False
    if weight_idx == 2:
        bias_meta_value = linear_node.args[0].meta.get('val')
        if bias_meta_value is None or meta_value.device.type != 'cpu' or bias_meta_value.dim() != 1 or (bias_meta_value.size(0) != weight_meta_value.size(1)):
            return False
    if input_meta_value.dtype == torch.bfloat16 or weight_meta_value.dtype == torch.bfloat16:
        if not mkldnn._is_mkldnn_bf16_supported():
            return False
    return True