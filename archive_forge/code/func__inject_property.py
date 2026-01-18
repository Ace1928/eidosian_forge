import torch
from torch.nn.modules.container import ModuleList, ModuleDict, Module
from torch.nn.parameter import Parameter
from torch import Tensor
import collections
import copyreg
from copy import deepcopy
from contextlib import contextmanager
from typing import Union, Optional, Dict, Tuple, Sequence
def _inject_property(module: Module, tensor_name: str) -> None:
    """Injects a property into module[tensor_name].

    It assumes that the class in the module has already been modified from its
    original one using _inject_new_class and that the tensor under :attr:`tensor_name`
    has already been moved out

    Args:
        module (nn.Module): module into which to inject the property
        tensor_name (str): name of the name of the property to create
    """
    assert not hasattr(module, tensor_name)

    @torch.jit.unused
    def get_cached_parametrization(parametrization) -> Tensor:
        global _cache
        key = (id(module), tensor_name)
        tensor = _cache.get(key)
        if tensor is None:
            tensor = parametrization()
            _cache[key] = tensor
        return tensor

    def get_parametrized(self) -> Tensor:
        if torch.jit.is_scripting():
            raise RuntimeError('Parametrization is not working with scripting.')
        parametrization = self.parametrizations[tensor_name]
        if _cache_enabled:
            if torch.jit.is_scripting():
                raise RuntimeError('Caching is not implemented for scripting. Either disable caching or avoid scripting.')
            elif torch._C._get_tracing_state() is not None:
                raise RuntimeError('Cannot trace a model while caching parametrizations.')
            else:
                return get_cached_parametrization(parametrization)
        else:
            return parametrization()

    def set_original(self, value: Tensor) -> None:
        if torch.jit.is_scripting():
            raise RuntimeError('Parametrization is not working with scripting.')
        self.parametrizations[tensor_name].right_inverse(value)
    setattr(module.__class__, tensor_name, property(get_parametrized, set_original))