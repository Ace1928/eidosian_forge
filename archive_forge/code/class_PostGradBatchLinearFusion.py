import collections
import logging
import operator
from typing import Any, DefaultDict, Deque, Dict, Iterator, List, Optional, Set, Tuple
import torch
from torch._dynamo.utils import counters
from torch._utils_internal import print_graph
from .. import config
from ..pattern_matcher import (
@register_fusion('batch_linear_post_grad', pre_grad=False)
class PostGradBatchLinearFusion(BatchFusion):
    """
    Fuse ops in a batch way in post grad (aten level).
    """

    def _addmm_node_can_be_fused(self, node: torch.fx.Node) -> bool:
        return node.kwargs.get('beta', 1.0) == 1.0 and node.kwargs.get('alpha', 1.0) == 1.0

    def _is_input_2d(self, input: torch.fx.Node) -> bool:
        return len(input.meta['tensor_meta'].shape) == 2

    def match(self, node: torch.fx.Node) -> Optional[Tuple[str, int, int, int, bool]]:
        if CallFunctionVarArgs(aten.mm).match(node):
            input_m, weight_m = node.args
            bias_m = None
        elif CallFunctionVarArgs(aten.addmm.default).match(node) and self._addmm_node_can_be_fused(node):
            bias_m, input_m, weight_m = node.args
        else:
            return None
        if not self._is_input_2d(input_m) or not self._is_input_2d(weight_m):
            return None
        m, k = input_m.meta['tensor_meta'].shape
        n = weight_m.meta['tensor_meta'].shape[1]
        batch_key = ('batch_linear', m, k, n, bias_m is not None)
        return batch_key

    def fuse(self, graph: torch.fx.GraphModule, subset: List[torch.fx.Node]):
        batch_inputs = []
        batch_weights = []
        batch_biases = []
        batch_nodes = []
        for node in subset:
            if CallFunctionVarArgs(aten.addmm.default).match(node):
                bias, input, weight = node.args
            elif CallFunctionVarArgs(aten.mm.default).match(node):
                input, weight = node.args
                bias = None
            batch_nodes.append(node)
            batch_inputs.append(input)
            batch_weights.append(weight)
            batch_biases.append(bias)
        with graph.inserting_before(subset[-1]):
            fused_inputs = decompose_stack(graph, batch_inputs)
            fused_weights = decompose_stack(graph, batch_weights)
            fused_bmm = graph.call_function(aten.bmm, args=(fused_inputs, fused_weights))
        for i, original_mm in enumerate(batch_nodes):
            has_bias = False
            with graph.inserting_after(fused_bmm):
                new_mm = graph.call_function(aten.select, args=(fused_bmm, 0, i))
                if batch_biases[i]:
                    has_bias = True
                    new_bias_add = graph.call_function(aten.add, args=(batch_biases[i], new_mm))
            new_mm_cont = new_bias_add if has_bias else new_mm
            original_mm.replace_all_uses_with(new_mm_cont)
            new_mm_cont.meta.update(original_mm.meta)
            graph.erase_node(original_mm)