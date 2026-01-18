import contextlib
import copy
from abc import ABC, abstractmethod
from typing import (
import torch.nn as nn
def _run_policy(self, root_module: nn.Module, ignored_modules: Set[nn.Module], root_kwargs: Dict[str, Any]) -> Dict[nn.Module, Dict[str, Any]]:
    target_module_to_kwargs: Dict[nn.Module, Dict[str, Any]] = {}
    for module in root_module.modules():
        if module in ignored_modules:
            continue
        res = self._lambda_fn(module)
        if not isinstance(res, (dict, bool)):
            raise ValueError(f'The lambda_fn passed to CustomPolicy should return False/True or a kwarg dict, but it returned {res}')
        if not res:
            continue
        kwargs = copy.copy(root_kwargs)
        if isinstance(res, dict):
            kwargs.update(res)
        target_module_to_kwargs[module] = kwargs
    return target_module_to_kwargs