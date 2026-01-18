from contextlib import contextmanager, nullcontext
from typing import Any, Tuple
import torch
import torch.nn as nn
from torch.utils.checkpoint import (
from .contract import contract
@contextmanager
def _no_hook(module: nn.Module):
    """
    Disable hooks installed by checkpoint to avoid unintentional recursion
    during backward recomputation.
    """
    orig_enable_hook = checkpoint.state(module).enable_hook
    checkpoint.state(module).enable_hook = False
    try:
        yield
    finally:
        checkpoint.state(module).enable_hook = orig_enable_hook