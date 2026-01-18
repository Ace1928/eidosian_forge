import torch
from torch.ao.quantization.backend_config import BackendConfig
from torch.fx.graph import Node, Graph
from ..utils import _parent_name, NodePattern, Pattern
from ..fuser_method_mappings import get_fuser_method_new
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Union
from .custom_config import FuseCustomConfig
from .match_utils import MatchAllNode
from torch.nn.utils.parametrize import type_before_parametrizations
class FuseHandler(ABC):
    """ Base handler class for the fusion patterns
    """

    @abstractmethod
    def __init__(self, node: Node):
        pass

    @abstractmethod
    def fuse(self, load_arg: Callable, named_modules: Dict[str, torch.nn.Module], fused_graph: Graph, root_node: Node, extra_inputs: List[Any], matched_node_pattern: NodePattern, fuse_custom_config: FuseCustomConfig, fuser_method_mapping: Dict[Pattern, Union[torch.nn.Sequential, Callable]], is_qat: bool) -> Node:
        pass