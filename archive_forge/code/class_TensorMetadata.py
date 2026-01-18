import torch
import torch.fx
import traceback
from torch._dispatch.python import enable_python_dispatcher
from torch.fx.node import Node, map_aggregate
from typing import Any, Tuple, NamedTuple, Optional, Dict
from torch.fx._compatibility import compatibility
from torch._guards import detect_fake_mode
@compatibility(is_backward_compatible=True)
class TensorMetadata(NamedTuple):
    shape: torch.Size
    dtype: torch.dtype
    requires_grad: bool
    stride: Tuple[int, ...]
    memory_format: Optional[torch.memory_format]
    is_quantized: bool
    qparams: Dict[str, Any]