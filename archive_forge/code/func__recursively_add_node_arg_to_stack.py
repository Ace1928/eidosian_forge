import collections
import enum
import torch
from torch.fx import GraphModule
from torch.fx.graph import Graph, Node
from torch.ao.quantization.utils import getattr_from_fqn
from .ns_types import NSSubgraph, NSNodeTargetType
from .mappings import (
from .pattern_utils import (
from torch.ao.quantization import (
from typing import Dict, Tuple, List, Optional, Set, Any
def _recursively_add_node_arg_to_stack(self, arg: Any) -> None:
    """
        Adds all of the nodes in this arg to the stack, properly navigating
        through list, dicts and tuples.
        """
    if isinstance(arg, Node):
        self.stack.append(arg)
    elif isinstance(arg, torch.fx.immutable_collections.immutable_list) or type(arg) is tuple:
        for inner_arg in arg:
            self._recursively_add_node_arg_to_stack(inner_arg)
    elif isinstance(arg, torch.fx.immutable_collections.immutable_dict):
        for value in arg.values():
            self._recursively_add_node_arg_to_stack(value)