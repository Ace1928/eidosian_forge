import contextlib
import copy
from abc import ABC, abstractmethod
from typing import (
import torch.nn as nn
class CustomPolicy(_Policy):
    """
    This policy takes in a lambda function that maps a given ``nn.Module`` to
    either ``False``, ``True``, or a kwarg dictionary.
    - If the function returns ``False`` or an empty dictionary, then the module
      does not have the API applied.
    - If the function returns ``True``, then the module has the API applied
      with the root's kwargs.
    - If the function returns a non-empty dictionary, then the module has the
      API applied, and the dictionary overrides the root's kwargs.

    Example::

        >>> # xdoctest: +SKIP("undefined variables")
        >>> model = init_transformer_model(...)
        >>> def lambda_fn(module: nn.Module):
        >>>     if module is model.lm_head:
        >>>         return {"sharding_strategy": ShardingStrategy.SHARD_GRAD_OP}
        >>>     elif isinstance(module, TransformerBlock):
        >>>         return True
        >>>     return False
        >>> policy = CustomPolicy(lambda_fn)
        >>> fsdp_model = FSDP(model, auto_wrap_policy=policy)
    """

    def __init__(self, lambda_fn: Callable[[nn.Module], Union[bool, Dict[str, Any]]]):
        self._lambda_fn = lambda_fn

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