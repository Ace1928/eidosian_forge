import functools
import warnings
from typing import Any, Optional
import torch
from torch.types import _dtype
def _exit_autocast(mode):
    if torch._C._is_torch_function_mode_enabled():
        return torch.overrides.handle_torch_function(torch.amp._exit_autocast, [], mode)
    mode.__exit__(None, None, None)