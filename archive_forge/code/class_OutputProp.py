import torch
import torch.fx
from torch.fx import (
from torch.ao.ns.fx.utils import (
from torch.ao.ns.fx.ns_types import (
from torch.ao.ns.fx.graph_passes import _maybe_get_fqn
from torch.ao.quantization import QConfigMapping
from torch.ao.quantization.qconfig import QConfigAny
from torch.ao.quantization.utils import getattr_from_fqn
from torch.ao.quantization.fx.match_utils import _MatchResult
from torch.utils._pytree import tree_map
import collections
import copy
from typing import List, Dict, Set, Tuple, Callable, Any, Optional
import operator
class OutputProp:
    """
    Output propagation (modeled from shape propagation).

    Given a GraphModule and an example input, saves the output flowing
    through each node on `node.traced_result`.

    Code based on the example from
    https://pytorch.org/docs/stable/fx.html#the-interpreter-pattern
    """

    def __init__(self, mod):
        self.mod = mod
        self.graph = mod.graph
        self.modules = dict(self.mod.named_modules())

    def propagate(self, *args):
        args_iter = iter(args)
        env: Dict[str, Node] = {}

        def load_arg(a):
            return torch.fx.graph.map_arg(a, lambda n: env[n.name])

        def fetch_attr(target: str):
            target_atoms = target.split('.')
            attr_itr = self.mod
            for i, atom in enumerate(target_atoms):
                if not hasattr(attr_itr, atom):
                    raise RuntimeError(f'Node referenced nonexistent target {'.'.join(target_atoms[:i])}')
                attr_itr = getattr(attr_itr, atom)
            return attr_itr
        for node in self.graph.nodes:
            if node.op == 'placeholder':
                result = next(args_iter)
            elif node.op == 'get_attr':
                result = fetch_attr(node.target)
            elif node.op == 'call_function':
                result = node.target(*load_arg(node.args), **load_arg(node.kwargs))
            elif node.op == 'call_method':
                self_obj, *args = load_arg(node.args)
                kwargs = load_arg(node.kwargs)
                result = getattr(self_obj, node.target)(*args, **kwargs)
            elif node.op == 'call_module':
                result = self.modules[node.target](*load_arg(node.args), **load_arg(node.kwargs))
            if isinstance(result, torch.Tensor):
                node.traced_result = result
            env[node.name] = result
        return None