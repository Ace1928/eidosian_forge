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
def _remove_extra_dequantize(m: GraphModule):
    """
    Removes duplicate dequant nodes in the graph, for an operator that has
    multiple dequant nodes as a user, replace them with a single dequant node
    that can be shared across all the uses. This should be seen as the "reverse"
    of `_duplicate_dequantize_node`.
    """
    dq_op = torch.ops.quantized_decomposed.dequantize_per_tensor
    for n in m.graph.nodes:
        dq_users = [user for user in n.users if user.op == 'call_function' and user.target == dq_op]
        if len(dq_users) > 1:
            with m.graph.inserting_after(dq_users[0]):
                new_node = m.graph.create_node('call_function', dq_op, dq_users[0].args, {})
            for dq_user in dq_users:
                dq_user.replace_all_uses_with(new_node)
                m.graph.erase_node(dq_user)
    m.recompile()