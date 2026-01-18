from collections import defaultdict
from itertools import chain
import pickle
from typing import (
import torch
import torch.nn as nn
from torch.utils.hooks import RemovableHandle
from torch.utils._python_dispatch import TorchDispatchMode
def _create_backward_hook(self, name: str) -> Callable:
    """Insert the current module name with backward prefix for the operator name."""

    def _backward_hook(module: nn.Module, grad_input: torch.Tensor, grad_output: torch.Tensor) -> None:
        self._cur_module_name = f'{name}.backward'
    return _backward_hook