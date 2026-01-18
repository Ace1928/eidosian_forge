from contextlib import contextmanager
from dataclasses import asdict
from enum import Enum
from typing import Any
import torch
from torch import nn
from peft.tuners.tuners_utils import BaseTuner, BaseTunerLayer, check_target_module_exists
from peft.utils import (
from .config import PolyConfig
from .layer import Linear, PolyLayer
def _register_pre_hooks(self, task_ids):
    """Helper method to register pre hooks."""
    if task_ids is None:
        return []

    def pre_hook(_, args, kwargs):
        kwargs['task_ids'] = task_ids
        return (args, kwargs)
    handles = []
    for module in self.model.modules():
        if isinstance(module, Linear):
            handle = module.register_forward_pre_hook(pre_hook, with_kwargs=True)
            handles.append(handle)
    return handles