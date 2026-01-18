import copy
import operator
from copy import deepcopy
from typing import cast, Dict, List, Optional, Union
import torch
import torch.fx._pytree as fx_pytree
import torch.utils._pytree as pytree
from torch.export import ExportedProgram
from torch.export.exported_program import (
from torch.fx import GraphModule
from .utils import _check_input_constraints_pre_hook
def _sink_params(module: GraphModule, inputs_to_state: Dict[str, str], scope: List[str]):
    """Sink params and buffers from graph inputs into get_attr nodes.

    Exported modules are purely functional, so they pass their parameters and
    buffers in as inputs to the graph.

    To replicate eager's semantics, we need to get them from the module state
    via get_attr instead.

    module: GraphModule, potentially containining nested submodules.
    inputs_to_state: mapping graph input names to the corresponding key in the state_dict.
    scope: tracks where we are in the module hierarchy, so that we can emit the
        right `getattr(self, "foo.bar")` calls, etc.
    """
    for name, submodule in module._modules.items():
        _sink_params(cast(GraphModule, submodule), inputs_to_state, scope + [name])
    if not isinstance(module, GraphModule):
        return
    graph = module.graph
    inputs = filter(lambda n: n.op == 'placeholder', graph.nodes)
    call_module_nodes = filter(lambda n: n.op == 'call_module', graph.nodes)
    for node in call_module_nodes:
        node.args = tuple(filter(lambda n: n.name not in inputs_to_state, node.args))
    for node in inputs:
        if node.name not in inputs_to_state:
            continue
        if len(node.users) > 0:
            state_name = inputs_to_state[node.name].split('.')
            if state_name[:len(scope)] != scope:
                continue
            attr_path = state_name[len(scope):]
            state_attr = _recursive_getattr(module, attr_path)
            assert isinstance(state_attr, torch.Tensor)
            with graph.inserting_after(node):
                new_node = graph.create_node('get_attr', '.'.join(attr_path))
            node.replace_all_uses_with(new_node, propagate_meta=True)
        graph.erase_node(node)
    module.recompile()