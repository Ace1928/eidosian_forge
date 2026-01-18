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
class _FabricModule(_DeviceDtypeModuleMixin):

    def __init__(self, forward_module: nn.Module, precision: Precision, original_module: Optional[nn.Module]=None) -> None:
        """The FabricModule is a thin wrapper around the :class:`torch.nn.Module` and handles precision / autocast
        automatically for the forward pass.

        The underlying wrapped module can be accessed via the property :attr:`module`.

        Args:
            forward_module: The module to wrap the ``forward`` method on.
            precision: Reference to the precision plugin for handling precision context
            original_module: The original, unmodified module as passed into the
                :meth:`lightning_fabric.fabric.Fabric.setup` method. This is needed when attribute lookup
                on this wrapper should pass through to the original module.

        """
        super().__init__()
        self._forward_module = forward_module
        self._original_module = original_module or forward_module
        self._precision = precision
        self._fabric_module_initialized = True

    @property
    def module(self) -> nn.Module:
        return self._original_module or self._forward_module

    @override
    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Casts all inputs to the right precision and handles autocast for operations in the module forward method."""
        args, kwargs = self._precision.convert_input((args, kwargs))
        with self._precision.forward_context():
            output = self._forward_module(*args, **kwargs)
        output = self._precision.convert_output(output)
        return output

    @overload
    def state_dict(self, *, destination: T_destination, prefix: str=..., keep_vars: bool=...) -> T_destination:
        ...

    @overload
    def state_dict(self, *, prefix: str=..., keep_vars: bool=...) -> Dict[str, Any]:
        ...

    @override
    def state_dict(self, destination: Optional[T_destination]=None, prefix: str='', keep_vars: bool=False) -> Optional[Dict[str, Any]]:
        return self._original_module.state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)

    @override
    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool=True, **kwargs: Any) -> _IncompatibleKeys:
        return self._original_module.load_state_dict(state_dict=state_dict, strict=strict, **kwargs)

    def _redirection_through_forward(self, method_name: str) -> Callable:
        assert method_name != 'forward'
        original_forward = self._original_module.forward

        def wrapped_forward(*args: Any, **kwargs: Any) -> Any:
            self._original_module.forward = original_forward
            method = getattr(self._original_module, method_name)
            return method(*args, **kwargs)

        def call_forward_module(*args: Any, **kwargs: Any) -> Any:
            self._original_module.forward = wrapped_forward
            return self.forward(*args, **kwargs)
        return call_forward_module

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

    @override
    def __getattr__(self, item: Any) -> Any:
        if item in _LIGHTNING_MODULE_STEP_METHODS and self._forward_module != self._original_module:
            return self._redirection_through_forward(item)
        try:
            return super().__getattr__(item)
        except AttributeError:
            original_module = super().__getattr__('_original_module')
            attr = getattr(original_module, item)
            if inspect.ismethod(attr) and self._forward_module != self._original_module:
                attr = self._wrap_method_with_module_call_tracker(attr, item)
            return attr

    @override
    def __setattr__(self, name: str, value: Any) -> None:
        if not getattr(self, '_fabric_module_initialized', False):
            super().__setattr__(name, value)
            return
        original_module = self._original_module
        original_has_attr = hasattr(original_module, name)
        fabric_has_attr = name in dir(self)
        if not (original_has_attr or fabric_has_attr):
            setattr(original_module, name, value)
            return
        if original_has_attr:
            setattr(original_module, name, value)
        if fabric_has_attr:
            super().__setattr__(name, value)