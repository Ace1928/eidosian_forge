from typing import Any, Dict, List, NamedTuple, Optional
import torch
from torch.fx._compatibility import compatibility
from torch.fx.graph import Graph
from torch.fx.graph_module import GraphModule
from torch.fx.node import (
from torch.fx.passes.shape_prop import ShapeProp
@compatibility(is_backward_compatible=False)
class size_bytes(NamedTuple):
    output_size: int
    total_size: int