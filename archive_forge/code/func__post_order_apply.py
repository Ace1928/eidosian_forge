import contextlib
import copy
from abc import ABC, abstractmethod
from typing import (
import torch.nn as nn
def _post_order_apply(root_module: nn.Module, fn: Callable[[nn.Module], Optional[nn.Module]]):
    """
    This applies ``fn`` to every module in the module tree of ``root_module``
    following a post-order traversal. If ``fn`` returns an :class:`nn.Module`,
    then this replaces the original module with the newly returned one in the
    tree. Otherwise, ``fn`` should return ``None``, in which case the module is
    not changed.
    """
    visited_modules: Set[nn.Module] = {root_module}

    def _post_order_apply_inner(module: nn.Module, module_name: str, parent_module: Optional[nn.Module]):
        for child_module_name, child_module in module.named_children():
            if child_module not in visited_modules:
                visited_modules.add(child_module)
                _post_order_apply_inner(child_module, child_module_name, module)
        optional_module = fn(module)
        if optional_module is not None:
            assert isinstance(parent_module, nn.Module), f'Non-root modules should have their parent module set but got {parent_module} for {module}'
            assert module_name, f'Non-root modules should have their module name set but got an empty module name for {module}'
            assert isinstance(optional_module, nn.Module), f'fn should return None or an nn.Module but got {optional_module}'
            setattr(parent_module, module_name, optional_module)
    _post_order_apply_inner(root_module, '', None)