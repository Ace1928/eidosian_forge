import hashlib
import torch
import torch.fx
from typing import Any, Dict, Optional, TYPE_CHECKING
from torch.fx.node import _get_qualified_name, _format_arg
from torch.fx.graph import _parse_stack_trace
from torch.fx.passes.shape_prop import TensorMetadata
from torch.fx._compatibility import compatibility
from itertools import chain
@compatibility(is_backward_compatible=False)
class FxGraphDrawer:

    def __init__(self, graph_module: torch.fx.GraphModule, name: str, ignore_getattr: bool=False, parse_stack_trace: bool=False):
        raise RuntimeError('FXGraphDrawer requires the pydot package to be installed. Please install pydot through your favorite Python package manager.')