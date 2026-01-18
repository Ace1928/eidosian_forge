import contextlib
from typing import Optional, Union, List, Set, Dict, Any
import warnings
from dataclasses import dataclass
import torch
import torchgen
from torch._C import _len_torch_dispatch_stack, _get_dispatch_stack_at,\
def _pop_mode(k: Optional[Union[DispatchKey, torch._C._TorchDispatchModeKey]]=None):
    if k is None or isinstance(k, torch._C._TorchDispatchModeKey):
        return _pop_torch_dispatch_stack(k)
    from torch._ops import pop_mode_for_key
    return pop_mode_for_key(k)