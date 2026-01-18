import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple
import torch
import torch.fx
from torch.fx._compatibility import compatibility
from torch.fx.node import map_arg
from .shape_prop import ShapeProp
from .split_utils import split_by_tags
from .tools_common import (
def _binary_traverse(self, nodes: NodeList) -> NodeSet:
    """
        Binary search on `nodes` for culprit.
        """
    return self._binary_search_impl(nodes, 0, len(nodes))