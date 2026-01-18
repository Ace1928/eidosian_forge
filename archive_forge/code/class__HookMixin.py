import functools
import inspect
import warnings
from collections import OrderedDict
from typing import Any, List, Optional, Tuple
import torch
import torch._C as _C
import torch._functorch as _functorch
import torch.utils.hooks as hooks
from torch._C import _functions
from torch._functorch.autograd_function import custom_function_call
class _HookMixin:

    @staticmethod
    def _register_hook(backward_hooks, hook):
        if backward_hooks is None:
            backward_hooks = OrderedDict()
        handle = hooks.RemovableHandle(backward_hooks)
        backward_hooks[handle.id] = hook
        return (backward_hooks, handle)