import inspect
from copy import deepcopy
from functools import wraps
from typing import (
import torch
from lightning_utilities.core.apply_func import apply_to_collection
from torch import Tensor
from torch import nn as nn
from torch.nn.modules.module import _IncompatibleKeys
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from typing_extensions import override
from lightning_fabric.plugins import Precision
from lightning_fabric.strategies import Strategy
from lightning_fabric.utilities import move_data_to_device
from lightning_fabric.utilities.data import _set_sampler_epoch
from lightning_fabric.utilities.device_dtype_mixin import _DeviceDtypeModuleMixin
from lightning_fabric.utilities.imports import _TORCH_GREATER_EQUAL_2_0
from lightning_fabric.utilities.types import Optimizable
def _wrap_method_with_module_call_tracker(self, method: Callable, name: str) -> Callable:
    """Tracks whether any submodule in ``self._original_module`` was called during the execution of ``method`` by
        registering forward hooks on all submodules."""
    module_called = False

    def hook(*_: Any, **__: Any) -> None:
        nonlocal module_called
        module_called = True

    @wraps(method)
    def _wrapped_method(*args: Any, **kwargs: Any) -> Any:
        handles = []
        for module in self._original_module.modules():
            handles.append(module.register_forward_hook(hook))
        output = method(*args, **kwargs)
        if module_called:
            raise RuntimeError(f'You are calling the method `{type(self._original_module).__name__}.{name}()` from outside the model. This will bypass the wrapper from the strategy and result in incorrect behavior in `.backward()`. You should pass your inputs through `forward()`.')
        for handle in handles:
            handle.remove()
        return output
    return _wrapped_method