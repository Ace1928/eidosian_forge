import itertools
from typing import Any, Callable, Dict, Optional, Sequence
import torch
from torch.overrides import TorchFunctionMode
from typing_extensions import override
from lightning_fabric.utilities.imports import _TORCH_GREATER_EQUAL_2_1
from lightning_fabric.utilities.types import _DEVICE
def _materialize_meta_tensors(module: torch.nn.Module, device: _DEVICE) -> None:
    """Materialize all tensors in a given module."""
    for module in module.modules():
        if any((t.is_meta for t in itertools.chain(module.parameters(recurse=False), module.buffers(recurse=False)))):
            _materialize(module, device)