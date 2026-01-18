import abc
import typing as t
import torch
import torch.fx
from torch.fx._compatibility import compatibility
from .shape_prop import TensorMetadata
from .tools_common import get_node_target, CALLABLE_NODE_OPS
class FunctionalOperatorSupport(OperatorSupportBase):

    def is_node_supported(self, submodules: t.Mapping[str, torch.nn.Module], node: torch.fx.Node) -> bool:
        return is_node_supported(submodules, node)