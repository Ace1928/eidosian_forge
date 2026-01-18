import collections
import logging
import operator
from typing import Any, DefaultDict, Deque, Dict, Iterator, List, Optional, Set, Tuple
import torch
from torch._dynamo.utils import counters
from torch._utils_internal import print_graph
from .. import config
from ..pattern_matcher import (
@register_fusion('batch_linear_lhs')
class BatchLinearLHSFusion(BatchFusion):
    """
    Batch linear left-hand side fusion. This pass tries to fuse the following patterns:

        torch.nn.functional.linear(x, w1), linear(x, w2),... * linear(x, wn)
        -> torch.mm(x, torch.cat([w1, w2,... * wn]).transpose(0, 1))

    We have a separate pass to eliminate contiguous transpose in a generic way.
    """

    def match(self, node: torch.fx.Node) -> Optional[Tuple[str, bool, Any]]:
        if CallFunctionVarArgs(torch.nn.functional.linear).match(node) and is_linear_node_can_be_fused(node):
            input = get_arg_value(node, 0, 'input')
            bias = get_arg_value(node, 2, 'bias')
            group_key = ('batch_linear_lhs', bias is None, input)
        else:
            group_key = None
        return group_key

    def fuse(self, graph: torch.fx.GraphModule, subset: List[torch.fx.Node]):
        batch_nodes = []
        batch_input = None
        batch_weights = []
        batch_biases = []
        split_sections = []
        for node in subset:
            input = get_arg_value(node, 0, 'input')
            weight = get_arg_value(node, 1, 'weight')
            bias = get_arg_value(node, 2, 'bias')
            batch_nodes.append(node)
            if batch_input is None:
                batch_input = input
            else:
                assert batch_input is input
            batch_weights.append(weight)
            if bias:
                batch_biases.append(bias)
            split_sections.append(weight.meta['example_value'].shape[0])
        with graph.inserting_before(subset[0]):
            cat_weights = graph.call_function(torch.cat, args=(batch_weights,), kwargs={'dim': 0})
            transposed_weights = graph.call_function(torch.transpose, args=(cat_weights, 0, 1))
            if len(batch_biases) > 0:
                cat_biases = graph.call_function(torch.cat, args=(batch_biases,), kwargs={'dim': 0})
                fused_lhs = graph.call_function(torch.addmm, args=(cat_biases, batch_input, transposed_weights))
            else:
                fused_lhs = graph.call_function(torch.mm, args=(batch_input, transposed_weights))
            fused_lhs_list = graph.call_function(torch.split, args=(fused_lhs, split_sections), kwargs={'dim': 1})
        for i, node in enumerate(batch_nodes):
            with graph.inserting_after(fused_lhs_list):
                new_node = graph.call_function(operator.getitem, args=(fused_lhs_list, i))
            node.replace_all_uses_with(new_node)
            new_node.meta.update(node.meta)
            graph.erase_node(node)