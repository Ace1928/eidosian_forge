import contextlib
import copy
from abc import ABC, abstractmethod
from typing import (
import torch.nn as nn
class ModuleWrapPolicy(_Policy):
    """
    This policy applies to every module of the specified module classes,
    passing in the kwargs given to the root.
    """

    def __init__(self, module_classes: Iterable[Type[nn.Module]]):
        module_classes_set = set(module_classes)
        self._module_classes = module_classes_set
        self._module_classes_str = str(module_classes_set)

    def _run_policy(self, root_module: nn.Module, ignored_modules: Set[nn.Module], root_kwargs: Dict[str, Any]) -> Dict[nn.Module, Dict[str, Any]]:
        module_classes = tuple(self._module_classes)
        target_module_to_kwargs: Dict[nn.Module, Dict[str, Any]] = {}
        for module in root_module.modules():
            if module in ignored_modules:
                continue
            elif isinstance(module, module_classes):
                target_module_to_kwargs[module] = copy.copy(root_kwargs)
        return target_module_to_kwargs

    def __call__(self, module, recurse, *args, **kwargs):
        return _module_wrap_policy(module, recurse, nonwrapped_numel=-1, module_classes=self._module_classes)

    def __repr__(self) -> str:
        return super().__repr__() + f'({self._module_classes_str})'