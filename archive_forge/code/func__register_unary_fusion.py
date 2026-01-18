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
def _register_unary_fusion():
    computation_call_fns = [_conv_call, _linear_call, _conv_transpose_call]

    def _unary_fusion_patterns(is_bf16):
        replacement_unary_fusion_patterns = {UnaryAttr('gelu', algorithm_attr='tanh'): [_unary_fusion_pattern(_gelu_fusion_2, call_fn, 4, is_bf16) for call_fn in computation_call_fns], UnaryAttr('gelu', algorithm_attr='none'): [_unary_fusion_pattern(_gelu_fusion_1, call_fn, 2, is_bf16) for call_fn in computation_call_fns], UnaryAttr('hardswish'): [_unary_fusion_pattern(_hardswish_fusion, call_fn, 2, is_bf16) for call_fn in computation_call_fns], UnaryAttr('hardsigmoid'): [_unary_fusion_pattern(_hardsigmoid_fusion, call_fn, 1, is_bf16) for call_fn in computation_call_fns], UnaryAttr('swish'): [_unary_fusion_pattern(_silu_fusion, call_fn, 2, is_bf16) for call_fn in computation_call_fns]}
        if not is_bf16:
            call_user1 = [call_fn(users=1) for call_fn in computation_call_fns]
            replacement_unary_fusion_patterns.update({UnaryAttr('relu'): [_combined_fusion(u, aten.relu) for u in call_user1], UnaryAttr('sigmoid'): [_combined_fusion(u, aten.sigmoid) for u in call_user1], UnaryAttr('tanh'): [_combined_fusion(u, aten.tanh) for u in call_user1]})
        return replacement_unary_fusion_patterns
    for is_bf16 in [True, False]:
        replace_patterns = _unary_fusion_patterns(is_bf16)
        for unary_attr, patterns in replace_patterns.items():
            _register_unary_fusion_lowering(patterns[0], unary_attr, computation_ops[0], is_bf16)
            _register_unary_fusion_lowering(patterns[1], unary_attr, computation_ops[1], is_bf16)
            _register_unary_fusion_lowering(patterns[2], unary_attr, computation_ops[2], is_bf16)
        _leaky_relu_patterns = [_unary_fusion_pattern(_leaky_relu_fusion, call_fn, 3, is_bf16) for call_fn in computation_call_fns]
        for pattern, computation_op in zip(_leaky_relu_patterns, computation_ops):
            _register_leaky_relu_fusion_lowering(pattern, computation_op, is_bf16)
        hardtanh_patterns = [_unary_fusion_pattern(_hardtanh_fusion, call_fn, 1, is_bf16) for call_fn in computation_call_fns]
        for pattern, computation_op in zip(hardtanh_patterns, computation_ops):
            _register_hardtanh_fusion_lowering(pattern, computation_op, is_bf16)