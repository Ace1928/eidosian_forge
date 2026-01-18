import contextlib
import copy
from abc import ABC, abstractmethod
from typing import (
import torch.nn as nn
def _construct_wrap_fn(root_module: nn.Module, target_module_to_kwargs: Dict[nn.Module, Dict[str, Any]], fsdp_fn: Callable) -> Callable[[nn.Module], Optional[nn.Module]]:
    """
    This constructs the "wrap" function to pass to :func:`_post_order_apply`
    based on ``target_module_to_kwargs``, which should be constructed from the
    wrapping policy.
    """

    def fn(module: nn.Module) -> Optional[nn.Module]:
        if module in target_module_to_kwargs and module is not root_module:
            kwargs = target_module_to_kwargs[module]
            return fsdp_fn(module, **kwargs)
        return None
    return fn