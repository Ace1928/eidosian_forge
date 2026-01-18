import abc
import typing as t
import torch
import torch.fx
from torch.fx._compatibility import compatibility
from .shape_prop import TensorMetadata
from .tools_common import get_node_target, CALLABLE_NODE_OPS
def _decline_if_node_in_names(submodules: t.Mapping[str, torch.nn.Module], node: torch.fx.Node) -> bool:
    if node.name in disallow_set:
        return False
    else:
        return True