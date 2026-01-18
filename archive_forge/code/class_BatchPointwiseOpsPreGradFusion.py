import collections
import logging
import operator
from typing import Any, DefaultDict, Deque, Dict, Iterator, List, Optional, Set, Tuple
import torch
from torch._dynamo.utils import counters
from torch._utils_internal import print_graph
from .. import config
from ..pattern_matcher import (
class BatchPointwiseOpsPreGradFusion(BatchPointwiseOpsFusionFactory):
    """
    Batch poinwise ops (e.g., sigmoid, relu, tanh) fusion in pre grad pass.
    We fuse it in random place, and the introduced stack node may be merged in split cat.
    """

    def __init__(self, op, **kwargs):
        super().__init__(op, **kwargs)
        self.op = op

    def match(self, node: torch.fx.Node):
        input = get_arg_value(node, 0, 'input')
        if CallFunctionVarArgs(self.op).match(node) and is_node_meta_valid(node):
            group_key = ('batch_' + self.op.__name__.lower() + '_pre_grad', str(input.meta['example_value'].shape), str(node.kwargs.get('inplace', False)))
        else:
            group_key = None
        return group_key

    def fuse(self, graph: torch.fx.GraphModule, subset: List[torch.fx.Node]):
        batch_nodes = []
        batch_inputs = []
        for node in subset:
            batch_nodes.append(node)
            batch_inputs.append(get_arg_value(node, 0, 'input'))
        with graph.inserting_before(subset[0]):
            stack_inputs = graph.call_function(torch.stack, args=(batch_inputs,), kwargs={'dim': 0})
            if self.op == torch.nn.functional.relu:
                batch_op = graph.call_function(self.op, args=(stack_inputs,), kwargs={'inplace': subset[0].kwargs.get('inplace', False)})
            else:
                batch_op = graph.call_function(self.op, args=(stack_inputs,))
            unbind_op = graph.call_function(torch.unbind, args=(batch_op,), kwargs={'dim': 0})
            for i, node in enumerate(batch_nodes):
                with graph.inserting_after(unbind_op):
                    getitem = graph.call_function(operator.getitem, args=(unbind_op, i))
                node.replace_all_uses_with(getitem)
                getitem.meta.update(node.meta)
                graph.erase_node(node)