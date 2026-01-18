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
def _register_qlinear_weight_prepack_pass(pattern, pass_number, dtype=torch.float32):

    @register_freezing_graph_pattern(pattern, extra_check=_is_valid_dequant_linear_pattern(dtype), pass_number=pass_number)
    def qlinear_weight_prepack(match: Match, *args, **kwargs):
        """
        Match the pattern:
        int8 activation
          |
        dequant_per_tensor
          |
        mm/addmm <- t <- dequant_per_channel <- int8_weight

        Insert weight prepack node and change the pattern to:
        int8 activation
          |
        onednn.qlinear_pointwise <- onednn.qlinear_prepack <- int8_weight
        """
        assert dtype in [torch.float32, torch.bfloat16]
        linear_node = match.output_node()
        assert linear_node.target in (aten.addmm.default, aten.mm.default)
        input_index = 0 if linear_node.target is aten.mm.default else 1
        weight_index = input_index + 1
        if dtype == torch.float32:
            mul_node = linear_node.args[input_index]
        else:
            activation_to_bf16_node = linear_node.args[input_index]
            mul_node = activation_to_bf16_node.args[0]
        sub_node = mul_node.args[0]
        to_fp32_node = sub_node.args[0]
        t_node = linear_node.args[weight_index]
        if dtype == torch.float32:
            dequant_per_channel = t_node.args[0]
        else:
            weight_to_bf16_node = t_node.args[0]
            dequant_per_channel = weight_to_bf16_node.args[0]
        assert dequant_per_channel.target is quantized_decomposed.dequantize_per_channel.default
        qx, x_zp, x_scale = (kwargs['x'], kwargs['x_zp'], kwargs['x_scale'])
        qw, w_scale, w_zp = (kwargs['q_weight'], kwargs['w_scale'], kwargs['w_zp'])
        bias = kwargs['b'] if 'b' in kwargs else None
        x_shape = qx.meta.get('tensor_meta').shape
        if has_free_symbols(x_shape):
            x_shape = None
        graph = match.graph
        with graph.inserting_before(linear_node):
            packed_weight_inputs = (qw, x_shape)
            packed_weight_op = torch.ops.onednn.qlinear_prepack
            prepack_weight_node = graph.call_function(packed_weight_op, args=packed_weight_inputs)
            new_args: Tuple[Any, ...] = (qx, x_scale, x_zp, prepack_weight_node, w_scale, w_zp, bias, 1.0, 0, dtype, 'none', [], '')
            new_linear_node = graph.call_function(torch.ops.onednn.qlinear_pointwise.default, args=new_args)
            linear_node.replace_all_uses_with(new_linear_node)
            new_linear_node.meta.update(linear_node.meta)
            graph.erase_node(linear_node)
            if dtype == torch.bfloat16:
                graph.erase_node(activation_to_bf16_node)
            graph.erase_node(mul_node)
            graph.erase_node(sub_node)
            graph.erase_node(to_fp32_node)
            graph.erase_node(t_node)
            if dtype == torch.bfloat16:
                graph.erase_node(weight_to_bf16_node)
            graph.erase_node(dequant_per_channel)
            counters['inductor']['qlinear_weight_prepack_matcher_count'] += 1
            counters['inductor']['qlinear_weight_prepack_matcher_nodes'] += len(match.nodes)