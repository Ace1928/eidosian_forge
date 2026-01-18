import dataclasses
import itertools
import operator
from typing import Any, Callable, Dict, List, Tuple
import torch
from torch.fx import Graph, GraphModule, Node
from torch.fx.subgraph_rewriter import (
import torch.nn.functional as F
from torch.ao.quantization.fx._decomposed import quantized_decomposed_lib  # noqa: F401
from torch.ao.quantization.quantizer import (
from .utils import (
def _fuse_conv_bn_qat_helper(m: GraphModule, conv_fn: Callable, example_inputs: Tuple[Any, ...], is_cuda: bool) -> GraphModule:
    """
    Given a graph of decomposed aten ops, replace the (conv + bn) pattern with
    the fused QAT subgraph equivalent. The input graph should already be annotated.
    The annotations in the original nodes will be preserved in the corresponding
    nodes in the new subgraph.

    Note: This also handles the (conv + bn + relu) pattern.
    """
    m.graph.eliminate_dead_code()
    m.recompile()
    conv_bn_pattern = _get_conv_bn_pattern(conv_fn)
    match_pattern = get_aten_graph_module(conv_bn_pattern, example_inputs, is_cuda)
    qat_conv_bn_pattern = _get_qat_conv_bn_pattern(conv_fn)
    replacement_pattern_with_conv_bias = get_aten_graph_module(qat_conv_bn_pattern, example_inputs, is_cuda)
    replacements_with_conv_bias = replace_pattern_with_filters(m, match_pattern, replacement_pattern_with_conv_bias, match_filters=[_has_conv_bias_filter], ignore_literals=True)
    m.recompile()
    qat_conv_bn_pattern_no_conv_bias = _get_qat_conv_bn_pattern_no_conv_bias(conv_fn)
    replacement_pattern_no_conv_bias = get_aten_graph_module(qat_conv_bn_pattern_no_conv_bias, example_inputs, is_cuda)
    replacements_no_conv_bias = replace_pattern_with_filters(m, match_pattern, replacement_pattern_no_conv_bias, match_filters=[_no_conv_bias_filter], ignore_literals=True)
    m.recompile()
    all_original_to_replacement_nodes = {}
    for r in replacements_with_conv_bias + replacements_no_conv_bias:
        for original_node, replacement_node in _get_conv_bn_pattern_nodes(r).values():
            replacement_node.meta = original_node.meta
            if _is_conv(original_node):
                _copy_over_literal_conv_args(original_node, replacement_node)
                _update_conv_input_qspec_map_after_replacement(original_node, replacement_node)
            all_original_to_replacement_nodes[original_node] = replacement_node
    for n in m.graph.nodes:
        _update_special_qspecs_after_replacement(n, all_original_to_replacement_nodes)
    return m