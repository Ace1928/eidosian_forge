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
def _register_weight_pack_pass():

    @register_freezing_graph_pattern(CallFunction(aten.convolution.default, *_aten_conv_args), extra_check=_is_packable_convolution)
    def convolution(match, *args, **kwargs):
        is_transposed = kwargs.get('is_transposed')
        assert isinstance(is_transposed, bool)
        graph = match.graph
        conv_node = match.output_node()
        input_size = conv_node.args[0].meta.get('val').shape
        with graph.inserting_before(conv_node):
            constant_args = [args[4], args[3], args[5], args[-1]]
            packed_weight_op = mkldnn._reorder_convolution_weight
            packed_conv_op = mkldnn._convolution_pointwise.default
            if is_transposed:
                constant_args.insert(1, args[-2])
                packed_weight_op = mkldnn._reorder_convolution_transpose_weight
                packed_conv_op = mkldnn._convolution_transpose_pointwise.default
            if not has_free_symbols(input_size):
                packed_weight_inputs = (args[1],) + tuple(constant_args) + (input_size,)
                packed_weight_node = graph.create_node('call_function', packed_weight_op, args=packed_weight_inputs)
            else:
                assert not is_transposed
                packed_weight_node = args[1]
            packed_conv_inputs = (args[0], packed_weight_node, args[2]) + tuple(constant_args) + ('none', [], '')
            packed_conv_node = graph.create_node('call_function', packed_conv_op, tuple(packed_conv_inputs))
            conv_node.replace_all_uses_with(packed_conv_node)
            packed_conv_node.meta.update(conv_node.meta)
            graph.erase_node(conv_node)

    @register_freezing_graph_pattern(CallFunction(aten.mkldnn_rnn_layer.default, *_aten_mkldnn_rnn_layer_args), extra_check=_is_packable_mkldnn_rnn_layer)
    def mkldnn_rnn_layer(match, *args, **kwargs):

        def get_item(graph, node, index):
            return graph.call_function(operator.getitem, (node, index))
        graph = match.graph
        lstm_node = match.output_node()
        input = args[0]
        weight0, weight1 = args[1:3]
        reverse = kwargs.get('reverse')
        packed_lstm_op = aten.mkldnn_rnn_layer.default
        hidden_size = args[9]
        has_biases = args[11]
        batch_first = args[13]
        with graph.inserting_before(lstm_node):
            packed_weight_op = mkldnn._reorder_mkldnn_rnn_layer_weight.default
            packed_weight_inputs = (weight0, weight1, hidden_size, reverse, has_biases, batch_first)
            packed_weight_node = graph.create_node('call_function', packed_weight_op, packed_weight_inputs, {}, 'name')
            packed_weight_items = [get_item(graph, packed_weight_node, i) for i in range(2)]
            pack_lstm_inputs = (args[0], *packed_weight_items, args[3], args[4], args[5], args[6], reverse, *args[7:])
            packed_lstm_node = graph.create_node('call_function', packed_lstm_op, args=pack_lstm_inputs)
            lstm_node.replace_all_uses_with(packed_lstm_node)
            packed_lstm_node.meta.update(lstm_node.meta)
            graph.erase_node(lstm_node)

    @register_freezing_graph_pattern(CallFunction(aten.addmm.default, Arg(), Arg(), Arg()), extra_check=_is_packable_linear)
    @register_freezing_graph_pattern(CallFunction(aten.mm.default, Arg(), Arg()), extra_check=_is_packable_linear)
    def linear(match, *args, **kwargs):
        graph = match.graph
        linear_node = match.output_node()
        input = args[0] if linear_node.target == aten.mm.default else args[1]
        bias = None if linear_node.target == aten.mm.default else args[0]
        weight = args[1] if linear_node.target == aten.mm.default else args[2]
        with graph.inserting_before(linear_node):
            transpose_weight_node = graph.create_node('call_function', aten.permute.default, (weight, (1, 0)))
            weight_dtype = weight.meta.get('val').dtype
            is_bf16_weight = weight_dtype == torch.bfloat16
            batch_size = input.meta.get('val').shape[0]
            if has_free_symbols(batch_size):
                assert is_bf16_weight, f'only bf16 weight prepacking supports dynamic shape inputs but got {weight_dtype}'
            packed_weight_inputs = (transpose_weight_node, batch_size.node.shape_env.size_hint(batch_size.node.expr) if has_free_symbols(batch_size) else batch_size)
            packed_weight_op = mkldnn._reorder_linear_weight if is_bf16_weight else torch.ops.mkl._mkl_reorder_linear_weight
            packed_weight_node = graph.create_node('call_function', packed_weight_op, args=packed_weight_inputs)
            packed_linear_inputs: Tuple[Any, ...] = (input, packed_weight_node)
            if is_bf16_weight:
                packed_linear_inputs += (bias, 'none', [], '')
                packed_linear_op = mkldnn._linear_pointwise.default
            else:
                packed_linear_inputs += (transpose_weight_node, bias, batch_size)
                packed_linear_op = torch.ops.mkl._mkl_linear
            packed_linear_node = graph.create_node('call_function', packed_linear_op, packed_linear_inputs)
            linear_node.replace_all_uses_with(packed_linear_node)
            packed_linear_node.meta.update(linear_node.meta)
            graph.erase_node(linear_node)