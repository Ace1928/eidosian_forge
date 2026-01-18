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
def _register_inplace_fusion():
    binary_ops = [aten.add, ops.add]
    inplace_fusion_op = mkldnn._convolution_pointwise_.binary
    outplace_fusion_op = mkldnn._convolution_pointwise.binary
    conv_call = _conv_call(users=1)
    conv_op = computation_ops[0]
    for binary_op in binary_ops:
        binary_v1 = _binary_fusion_v1(conv_call, binary_op)
        binary_unary_v1 = _combined_fusion(binary_v1, aten.relu)
        _register_binary_unary_maybe_inplace_fusion_lowering(binary_unary_v1, conv_op, binary_op, inplace_fusion_op, outplace_fusion_op, other_index=0, unary_attr=UnaryAttr('relu'))
        _register_binary_unary_maybe_inplace_fusion_lowering(binary_v1, conv_op, binary_op, inplace_fusion_op, outplace_fusion_op, other_index=0)
        binary_v2 = _binary_fusion_v2(conv_call, binary_op)
        binary_unary_v2 = _combined_fusion(binary_v2, aten.relu)
        _register_binary_unary_maybe_inplace_fusion_lowering(binary_unary_v2, conv_op, binary_op, inplace_fusion_op, outplace_fusion_op, other_index=1, unary_attr=UnaryAttr('relu'))
        _register_binary_unary_maybe_inplace_fusion_lowering(binary_v2, conv_op, binary_op, inplace_fusion_op, outplace_fusion_op, other_index=1)