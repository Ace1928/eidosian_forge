from collections import defaultdict
from itertools import chain
import pickle
from typing import (
import torch
import torch.nn as nn
from torch.utils.hooks import RemovableHandle
from torch.utils._python_dispatch import TorchDispatchMode
def _create_pre_forward_hook(self, name: str) -> Callable:
    """Prefix operator name with current module and 'forward', and insert 'fw_start' marker at forward pass start."""

    def _pre_forward_hook(module: nn.Module, inputs: Any) -> None:
        self._cur_module_name = f'{name}.forward'
        if hasattr(module, '_memory_tracker_is_root') and module._memory_tracker_is_root:
            self._add_marker('fw_start')
    return _pre_forward_hook