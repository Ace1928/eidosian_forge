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
def _register_quantized_conv_lowering(pattern, pass_number, computation_op, output_dtype, unary_attr, original_pattern_output_dtype=torch.float32):

    @register_lowering_pattern(pattern, extra_check=_is_valid_quantized_conv2d_optimization_pattern(output_dtype), pass_number=pass_number)
    def qconv(match: Match, *args, **kwargs):
        x, x_scale, x_zp = (kwargs['x'], kwargs['x_scale'], kwargs['x_zp'])
        packed_weight, w_scale, w_zp = (kwargs['packed_weight'], kwargs['w_scale'], kwargs['w_zp'])
        b, stride, padding, dilation, groups = (kwargs['b'], kwargs['stride'], kwargs['padding'], kwargs['dilation'], kwargs['groups'])
        assert output_dtype in [None, torch.float32, torch.bfloat16]
        o_inv_scale = kwargs['o_inv_scale'] if output_dtype is None else 1.0
        o_zero_point = kwargs['o_zp'] if output_dtype is None else 0
        assert kwargs['output_dtype'] is original_pattern_output_dtype
        assert kwargs['attr'] == 'none'
        if unary_attr.op_name == 'hardtanh':
            min_value = kwargs.get('min_value')
            max_value = kwargs.get('max_value')
            unary_attr.scalars_attr = [min_value, max_value]
        computation_args = (x, x_scale, x_zp, packed_weight, w_scale, w_zp, b, stride, padding, dilation, groups, o_inv_scale, o_zero_point, output_dtype, unary_attr.op_name, unary_attr.scalars_attr, unary_attr.algorithm_attr)
        counters['inductor']['qconv2d_unary_matcher_count'] += 1
        counters['inductor']['qconv2d_unary_matcher_nodes'] += len(match.nodes)
        return L[computation_op](*computation_args)
    return qconv