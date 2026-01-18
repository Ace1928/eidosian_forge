import torch
from torch.fx import GraphModule, map_arg
from torch.fx.graph import Graph, Node
from torch.ao.quantization.fx.utils import get_new_attr_name_with_prefix
from .utils import (
from .ns_types import (
from torch.ao.ns.fx.mappings import (
from torch.ao.quantization.observer import _is_activation_post_process
from typing import Dict, Tuple, Callable, List, Any, Union, Optional, Set
def _insert_copy_of_node_a_after_input_node_c(input_node_c: Union[Node, List[Node]], input_node_c_2: Optional[Union[Node, List[Node]]], node_a: Node, gm_a: GraphModule, gm_b: GraphModule, node_name_prefix: str) -> Node:
    """
    Assume that node_a from graph_a has
      args (input, (input2)?, arg1, ...), and
      kwargs {kw0: kwarg0, ...}

    Note: input2 is optional. If it equals to None, we assume that the op
    has a single non-param input.  If it is specified, we assume that the op
    has two non-param inputs.

    Copies the underlying values of arg1..argn and kwarg0..kwargn into gm_b,
    and creates the corresponding nodes in graph_c. Note: observers are ignored,
    so if an arg is an observer we navigate up until we find a non-observer parent.

    If node_a is a call_module, points the module pointed to by node_a to gm_b.

    Creates the copy of node_a in graph_c, with input as the first arg,
    and all other args and kwargs pointing to the copies of the objects
    in gm_b created above.

    An example in pictures:

    graph A:
    ========

    input -------------> node_a
                         / / /
    (input_2)?----------/ / /
                         / /
    weight -> weight_obs  /
                         /
    bias ----------------

    graph C (derived from B):
    =========================

    input_node_c --> node_a_copy
                     / / /
    (input_node_c_2)? / /
                     / /
    weight_copy ----/ /
                     /
    bias_copy ------/
    """
    if isinstance(input_node_c, Node):
        graph_c = input_node_c.graph
    else:
        assert isinstance(input_node_c, list)
        graph_c = input_node_c[0].graph
    norm_args_kwargs = node_a.normalized_arguments(gm_a, normalize_to_only_use_kwargs=True)
    if norm_args_kwargs is not None:
        norm_args, norm_kwargs = norm_args_kwargs
    else:
        norm_args, norm_kwargs = (node_a.args, node_a.kwargs)
    new_args = []
    new_kwargs = {}

    def _copy_arg(arg):
        if isinstance(arg, Node):
            arg = return_first_non_observer_node(arg, gm_a)
            arg = _copy_node_from_a_to_c(arg, gm_a, gm_b, graph_c)
            return arg
        elif isinstance(arg, (int, float, torch.dtype)):
            return arg
        elif isinstance(kwarg_val, (list, tuple)):
            for el in kwarg_val:
                assert not isinstance(el, Node), 'handling of Node inside list is not implemented'
            return arg
        else:
            raise AssertionError(f'handling for kwarg of type {type(kwarg_val)} is not implemented')
    cur_idx = 0
    while cur_idx < len(norm_args):
        if cur_idx == 0:
            new_arg = input_node_c
        elif cur_idx == 1 and input_node_c_2 is not None:
            new_arg = input_node_c_2
        else:
            new_arg = _copy_arg(norm_args[cur_idx])
        new_args.append(new_arg)
        cur_idx += 1
    for kwarg_name, kwarg_val in norm_kwargs.items():
        if cur_idx == 0:
            new_kwargs[kwarg_name] = input_node_c
        elif cur_idx == 1 and input_node_c_2 is not None:
            new_kwargs[kwarg_name] = input_node_c_2
        else:
            new_kwargs[kwarg_name] = _copy_arg(kwarg_val)
        cur_idx += 1
    new_args = tuple(new_args)
    node_a_shadows_c_name = get_new_attr_name_with_prefix(node_name_prefix)(gm_b)
    if node_a.op == 'call_module':
        new_mod_copy_name = get_new_attr_name_with_prefix(node_name_prefix)(gm_b)
        assert isinstance(node_a.target, str)
        mod_a = getattr_from_fqn(gm_a, node_a.target)
        setattr(gm_b, new_mod_copy_name, mod_a)
        node_a_shadows_c = graph_c.create_node(node_a.op, new_mod_copy_name, new_args, new_kwargs, node_a_shadows_c_name)
        return node_a_shadows_c
    else:
        assert node_a.op in ('call_function', 'call_method')
        node_a_shadows_c = graph_c.create_node(node_a.op, node_a.target, new_args, new_kwargs, node_a_shadows_c_name)
        return node_a_shadows_c