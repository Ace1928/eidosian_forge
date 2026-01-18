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
def _is_packable_convolution(match):
    """
        Check if the node is supported for MKLDNN convolution.
        """
    conv_node = match.output_node()
    input_meta_value = conv_node.args[0].meta.get('val')
    weight_meta_value = conv_node.args[1].meta.get('val')
    if input_meta_value is None or weight_meta_value is None:
        return False
    input_size = input_meta_value.shape
    if conv_node.args[1].op != 'get_attr':
        return False
    for meta_value in [input_meta_value, weight_meta_value]:
        if meta_value is None or meta_value.device.type != 'cpu' or meta_value.dim() != 4:
            return False
    if input_meta_value.dtype == torch.bfloat16 or weight_meta_value.dtype == torch.bfloat16:
        if not mkldnn._is_mkldnn_bf16_supported():
            return False
    is_transposed = conv_node.args[-3]
    if is_transposed:
        if has_free_symbols(input_size):
            return False
        groups = conv_node.args[-1]
        in_channels = weight_meta_value.size(0)
        if groups > 1 and groups == in_channels:
            return False
        output_paddings = conv_node.args[-2]
        strides = conv_node.args[3]
        if any((output_padding >= stride for output_padding, stride in zip(output_paddings, strides))):
            return False
    return True