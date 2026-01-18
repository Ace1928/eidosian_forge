import functools
import warnings
from typing import Any, Optional
import torch
from torch.types import _dtype
def _enter_autocast(*vals):
    if torch._C._is_torch_function_mode_enabled():
        return torch.overrides.handle_torch_function(torch.amp._enter_autocast, [], *vals)
    mode = torch.amp.autocast(*vals)
    mode.__enter__()
    return mode