import hashlib
import torch
import torch.fx
from typing import Any, Dict, Optional, TYPE_CHECKING
from torch.fx.node import _get_qualified_name, _format_arg
from torch.fx.graph import _parse_stack_trace
from torch.fx.passes.shape_prop import TensorMetadata
from torch.fx._compatibility import compatibility
from itertools import chain
def _get_tensor_label(self, t: torch.Tensor) -> str:
    return str(t.dtype) + str(list(t.shape)) + '\\n'