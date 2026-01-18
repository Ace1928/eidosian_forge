from builtins import isinstance
import functools
import logging
from typing import Any, List, Tuple
import torch
from torch import nn
def _conditional_amp_fwd_decorator(orig_func):
    if hasattr(torch.cuda.amp, 'custom_fwd'):
        return torch.cuda.amp.custom_fwd(orig_func)

    @functools.wraps(orig_func)
    def inner_decorator(*args: Any, **kwargs: Any) -> Any:
        return orig_func(*args, **kwargs)
    return inner_decorator