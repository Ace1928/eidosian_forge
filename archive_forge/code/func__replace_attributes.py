from .graph_module import GraphModule
from .graph import Graph
from .node import Node
from ._symbolic_trace import symbolic_trace
from ._compatibility import compatibility
import copy
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Set, Union
import torch
def _replace_attributes(gm: GraphModule, replacement: torch.nn.Module) -> None:
    gm.delete_all_unused_submodules()
    if isinstance(replacement, GraphModule):
        replacement.graph.lint()

    def try_get_attr(gm: torch.nn.Module, target: str) -> Optional[Any]:
        module_path, _, attr_name = target.rpartition('.')
        mod: torch.nn.Module = gm.get_submodule(module_path)
        attr = getattr(mod, attr_name, None)
        return attr
    for node in gm.graph.nodes:
        if node.op == 'call_module' or node.op == 'get_attr':
            gm_attr = try_get_attr(gm, node.target)
            replacement_attr = try_get_attr(replacement, node.target)
            if gm_attr is not None:
                continue
            elif replacement_attr is not None:
                new_attr = copy.deepcopy(replacement_attr)
                if isinstance(replacement_attr, torch.nn.Module):
                    gm.add_submodule(node.target, new_attr)
                else:
                    setattr(gm, node.target, new_attr)
            else:
                raise RuntimeError('Attempted to create a "', node.op, f'" node during subgraph rewriting with target {node.target}, but the referenced attribute does not exist in the replacement GraphModule')
    gm.graph.lint()