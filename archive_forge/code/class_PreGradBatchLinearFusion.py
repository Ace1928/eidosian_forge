import collections
import logging
import operator
from typing import Any, DefaultDict, Deque, Dict, Iterator, List, Optional, Set, Tuple
import torch
from torch._dynamo.utils import counters
from torch._utils_internal import print_graph
from .. import config
from ..pattern_matcher import (
@register_fusion('batch_linear')
class PreGradBatchLinearFusion(BatchFusion):
    """
    Batch linear fusion in pre grad pass.
    Fuse linear with same size with torch.baddmm
    """

    def _getitem_args(self, getitem_node: torch.fx.Node):
        if getitem_node.target != operator.__getitem__ or getitem_node.op != 'call_function':
            return None
        return getitem_node.args[0]

    def match(self, node: torch.fx.Node):
        if CallFunctionVarArgs(torch.nn.functional.linear).match(node) and is_linear_node_can_be_fused(node):
            input = get_arg_value(node, 0, 'input')
            weight = get_arg_value(node, 1, 'weight')
            bias = get_arg_value(node, 2, 'bias')
            group_key = ('batch_linear_pre_grad', self._getitem_args(input), str(input.meta['example_value'].shape), str(weight.meta['example_value'].shape), bias is None)
        else:
            group_key = None
        return group_key

    def fuse(self, graph: torch.fx.GraphModule, subset: List[torch.fx.Node]):
        batch_nodes = []
        batch_inputs = []
        batch_weights = []
        batch_biases = []
        for node in subset:
            batch_nodes.append(node)
            batch_inputs.append(get_arg_value(node, 0, 'input'))
            batch_weights.append(get_arg_value(node, 1, 'weight'))
            batch_biases.append(get_arg_value(node, 2, 'bias'))
        with graph.inserting_before(subset[0]):
            stack_inputs = graph.call_function(torch.stack, args=(batch_inputs,), kwargs={'dim': 0})
            stack_weights = graph.call_function(torch.stack, args=(batch_weights,), kwargs={'dim': 0})
            transpose_weight = graph.call_function(torch.transpose, args=(stack_weights, 1, 2))
            if all((bias is None for bias in batch_biases)):
                bmm = graph.call_function(torch.bmm, args=(stack_inputs, transpose_weight))
            else:
                stack_biases = graph.call_function(torch.stack, args=(batch_biases,), kwargs={'dim': 0})
                unsqueeze_biases = graph.call_function(torch.unsqueeze, args=(stack_biases, 1))
                bmm = graph.call_function(torch.baddbmm, args=(unsqueeze_biases, stack_inputs, transpose_weight))
            bmm = graph.call_function(torch.unbind, args=(bmm,), kwargs={'dim': 0})
            for i, linear in enumerate(batch_nodes):
                with graph.inserting_after(bmm):
                    getitem = graph.call_function(operator.getitem, args=(bmm, i))
                linear.replace_all_uses_with(getitem)
                getitem.meta.update(linear.meta)
                graph.erase_node(linear)