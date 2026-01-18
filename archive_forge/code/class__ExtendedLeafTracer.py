import logging
from typing import Dict, List, Set
import torch
import torch.fx
from torch.fx.node import Node
class _ExtendedLeafTracer(torch.fx.Tracer):
    """Tracer with an extended set of leaf nn.Modules."""

    def __init__(self, leaf_modules: Set[torch.nn.Module]):
        """Initializes a new _ExtendedLeafTracer object.

        Args:
            leaf_modules: The set of extra nn.Modules instances which will not be traced
                through but instead considered to be leaves.
        """
        super().__init__()
        self.leaf_modules = leaf_modules

    def is_leaf_module(self, m: torch.nn.Module, model_qualified_name: str) -> bool:
        return super().is_leaf_module(m, model_qualified_name) or m in self.leaf_modules