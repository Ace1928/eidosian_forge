import warnings
from enum import auto, Enum
from functools import partial
from typing import Any, Callable, Dict, Iterator, Optional, Tuple
import torch
import torch.nn as nn
from torch.autograd.graph import save_on_cpu
from torch.distributed.utils import _pack_kwargs, _replace_by_prefix, _unpack_kwargs
from torch.utils.checkpoint import checkpoint as torch_utils_checkpoint
class OffloadWrapper(ActivationWrapper):

    def __init__(self, mod):
        super().__init__(mod)

    def forward(self, *args, **kwargs):
        with save_on_cpu(pin_memory=True):
            return self._checkpoint_wrapped_module(*args, **kwargs)