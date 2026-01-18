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
def _register_quantization_binary_fusion():

    class BinaryUnaryAttr:

        def __init__(self, binary_op_name: str, alpha=None, unary_op_name: str='none', scalars_attr=None, algorithm_attr=None):
            self.binary_op_name = binary_op_name
            self.alpha = alpha if alpha else 1.0
            self.unary_op_name = unary_op_name
            self.scalars_attr = scalars_attr if scalars_attr else []
            self.algorithm_attr = algorithm_attr if algorithm_attr else ''
    for int8_mixed_bf16_with_inplace_add in [False, True]:
        binary_replace_patterns = {BinaryUnaryAttr('add', 1.0, 'none', [], ''): generate_pattern_with_output_quant(generate_pattern_with_binary(aten.add.Tensor, dequantize_qconv_pt2e_pattern, dequantize_accum_pattern, int8_mixed_bf16_with_inplace_add), dtype=torch.bfloat16 if int8_mixed_bf16_with_inplace_add else torch.float32), BinaryUnaryAttr('add', 1.0, 'relu', [], ''): generate_pattern_with_output_quant(generate_pattern_with_unary(generate_pattern_with_binary(aten.add.Tensor, dequantize_qconv_pt2e_pattern, dequantize_accum_pattern, int8_mixed_bf16_with_inplace_add), aten.relu.default), dtype=torch.bfloat16 if int8_mixed_bf16_with_inplace_add else torch.float32)}
        for binary_unary_attr, patterns in binary_replace_patterns.items():
            _register_quantized_conv_binary_lowering(patterns, 0, torch.ops.onednn.qconv2d_pointwise.binary, None, binary_unary_attr)
        binary_replace_float_out_patterns = {BinaryUnaryAttr('add', 1.0, 'relu', [], ''): generate_pattern_with_unary(generate_pattern_with_binary(aten.add.Tensor, dequantize_qconv_pt2e_pattern, KeywordArg('accum_after_dequant'), int8_mixed_bf16_with_inplace_add), aten.relu.default)}
        for binary_unary_attr, patterns in binary_replace_float_out_patterns.items():
            if int8_mixed_bf16_with_inplace_add:
                _register_quantized_conv_binary_lowering(patterns, 0, torch.ops.onednn.qconv2d_pointwise.binary, torch.bfloat16, binary_unary_attr)
            else:
                _register_quantized_conv_binary_lowering(patterns, 1, torch.ops.onednn.qconv2d_pointwise.binary, torch.float32, binary_unary_attr)
        binary_replace_float_out_patterns = {BinaryUnaryAttr('add', 1.0, 'none', [], ''): generate_pattern_with_binary(aten.add.Tensor, dequantize_qconv_pt2e_pattern, KeywordArg('accum_after_dequant'), int8_mixed_bf16_with_inplace_add)}
        for binary_unary_attr, patterns in binary_replace_float_out_patterns.items():
            _register_quantized_conv_binary_lowering(patterns, 1 if int8_mixed_bf16_with_inplace_add else 2, torch.ops.onednn.qconv2d_pointwise.binary, torch.bfloat16 if int8_mixed_bf16_with_inplace_add else torch.float32, binary_unary_attr)