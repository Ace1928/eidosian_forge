import functools
from typing import Dict, List, Mapping, Optional, Union
import torch
import torch.nn as nn
from .state import PartialState
from .utils import (
from .utils.modeling import get_non_persistent_buffers
from .utils.other import recursive_getattr
class UserCpuOffloadHook:
    """
    A simple hook grouping a model and a `ModelHook`, which provides easy APIs for to call the init method of the hook
    or remove it entirely.
    """

    def __init__(self, model, hook):
        self.model = model
        self.hook = hook

    def offload(self):
        self.hook.init_hook(self.model)

    def remove(self):
        remove_hook_from_module(self.model)