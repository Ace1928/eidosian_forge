import torch
from torch.fx import GraphModule, map_arg
from torch.fx.graph import Graph, Node
from torch.ao.quantization.fx.utils import get_new_attr_name_with_prefix
from .utils import (
from .ns_types import (
from torch.ao.ns.fx.mappings import (
from torch.ao.quantization.observer import _is_activation_post_process
from typing import Dict, Tuple, Callable, List, Any, Union, Optional, Set
def _can_insert_copy_of_subgraph_a(subgraph_a: NSSubgraph, gm_a: GraphModule, num_non_param_args_node_a: int) -> bool:
    """
    This function returns `False` if the input subgraph cannot be copied by
    `_insert_copy_of_subgraph_a_after_input_node_c`. This usually means
    that there is a corner case logic for which copy is not yet implemented.
    """
    nodes = []
    cur_node = subgraph_a.end_node
    while cur_node != subgraph_a.start_node:
        nodes.append(cur_node)
        cur_node = get_normalized_nth_input(cur_node, gm_a, 0)
    nodes.append(cur_node)
    nodes.reverse()

    def _can_insert(node_a_arg, gm_a):
        if isinstance(node_a_arg, Node):
            arg_a = return_first_non_observer_node(node_a_arg, gm_a)
            if arg_a.op == 'call_method':
                return arg_a.target in ('dequantize', 'to')
            elif arg_a.op == 'get_attr':
                return True
            else:
                return False
        elif isinstance(node_a_arg, (list, tuple)):
            for el in node_a_arg:
                if not isinstance(el, Node):
                    return False
        return True
    for node_a in nodes:
        local_num_non_param_args_node_a = num_non_param_args_node_a if node_a is nodes[0] else 1
        norm_args_kwargs = node_a.normalized_arguments(gm_a, normalize_to_only_use_kwargs=True)
        if norm_args_kwargs is not None:
            norm_args, norm_kwargs = norm_args_kwargs
        else:
            norm_args, norm_kwargs = (node_a.args, node_a.kwargs)
        cur_idx = 0
        while cur_idx < len(norm_args):
            if cur_idx == 0:
                pass
            elif cur_idx == 1 and local_num_non_param_args_node_a == 2:
                pass
            elif not _can_insert(norm_args[cur_idx], gm_a):
                return False
            cur_idx += 1
        for kwarg_val in norm_kwargs.values():
            if cur_idx == 0:
                pass
            elif cur_idx == 1 and local_num_non_param_args_node_a == 2:
                pass
            elif not _can_insert(kwarg_val, gm_a):
                return False
            cur_idx += 1
    return True