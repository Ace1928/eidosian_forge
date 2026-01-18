import abc
import typing as t
import torch
import torch.fx
from torch.fx._compatibility import compatibility
from .shape_prop import TensorMetadata
from .tools_common import get_node_target, CALLABLE_NODE_OPS
def _decline_if_input_dtype(submodules: t.Mapping[str, torch.nn.Module], node: torch.fx.Node) -> bool:
    for arg in node.all_input_nodes:
        if arg.op == 'get_attr':
            continue
        arg_dtype = _get_arg_dtype(arg)
        if arg_dtype == dtype:
            return False
    return True