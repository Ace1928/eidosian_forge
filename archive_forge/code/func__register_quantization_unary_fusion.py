import copy
import functools
import math
import operator
from typing import Any, Tuple
import torch
from torch._dynamo.utils import counters
from torch.fx.experimental.symbolic_shapes import has_free_symbols
from ..lowering import lowerings as L, require_channels_last
from ..pattern_matcher import Arg, CallFunction, filter_nodes, KeywordArg, ListOf, Match
from ..utils import pad_listlike
from .freezing_patterns import register_freezing_graph_pattern
from .post_grad import register_lowering_pattern
def _register_quantization_unary_fusion():

    class UnaryAttr:

        def __init__(self, op_name: str, scalars_attr=None, algorithm_attr=None):
            self.op_name = op_name
            self.scalars_attr = scalars_attr if scalars_attr else []
            self.algorithm_attr = algorithm_attr if algorithm_attr else ''
    for original_pattern_output_dtype in [torch.float32, torch.bfloat16]:
        conv_unary_replace_patterns = {UnaryAttr('none', [], ''): generate_pattern_with_output_quant(dequantize_qconv_pt2e_pattern, dtype=original_pattern_output_dtype), UnaryAttr('relu', [], ''): generate_pattern_with_output_quant(generate_pattern_with_unary(dequantize_qconv_pt2e_pattern, aten.relu.default), dtype=original_pattern_output_dtype), UnaryAttr('hardtanh', [], ''): generate_pattern_with_output_quant(generate_pattern_with_unary(dequantize_qconv_pt2e_pattern, aten.hardtanh.default), dtype=original_pattern_output_dtype)}
        for unary_attr, patterns in conv_unary_replace_patterns.items():
            _register_quantized_conv_lowering(patterns, 1, torch.ops.onednn.qconv2d_pointwise, None, unary_attr, original_pattern_output_dtype=original_pattern_output_dtype)
        conv_unary_replace_float_out_patterns = {UnaryAttr('relu', [], ''): generate_pattern_with_unary(dequantize_qconv_pt2e_pattern, aten.relu.default), UnaryAttr('hardtanh', [], ''): generate_pattern_with_unary(dequantize_qconv_pt2e_pattern, aten.hardtanh.default)}
        for unary_attr, patterns in conv_unary_replace_float_out_patterns.items():
            _register_quantized_conv_lowering(patterns, 2, torch.ops.onednn.qconv2d_pointwise, original_pattern_output_dtype, unary_attr, original_pattern_output_dtype=original_pattern_output_dtype)
        linear_unary_replace_patterns = {UnaryAttr('none', [], ''): generate_pattern_with_output_quant(qlinear_pt2e_pattern, dtype=original_pattern_output_dtype), UnaryAttr('relu', [], ''): generate_pattern_with_output_quant(generate_pattern_with_unary(qlinear_pt2e_pattern, aten.relu.default), dtype=original_pattern_output_dtype)}
        for unary_attr, patterns in linear_unary_replace_patterns.items():
            _register_quantized_linear_lowering(patterns, 1, torch.ops.onednn.qlinear_pointwise, None, unary_attr, original_pattern_output_dtype=original_pattern_output_dtype)
        linear_unary_replace_float_out_patterns = {UnaryAttr('relu', [], ''): generate_pattern_with_unary(qlinear_pt2e_pattern, aten.relu.default)}
        for unary_attr, patterns in linear_unary_replace_float_out_patterns.items():
            _register_quantized_linear_lowering(patterns, 2, torch.ops.onednn.qlinear_pointwise, original_pattern_output_dtype, unary_attr, original_pattern_output_dtype=original_pattern_output_dtype)