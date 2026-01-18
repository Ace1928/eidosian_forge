import collections
import functools
import inspect
import warnings
from functools import partial
from typing import Any, Callable, Dict, List, Set, Tuple, Type, Union
import torch.nn as nn
from torch.distributed.fsdp._common_utils import (
from torch.distributed.fsdp.wrap import (
def _validate_frozen_params(root_module: nn.Module, modules_to_wrap: Set[nn.Module], ignored_params: Set[nn.Parameter], use_orig_params: bool):
    """
    This checks that, given ``modules_to_wrap``, each module would manage
    parameters that are uniformly frozen or non-frozen. This uniformity
    requirement is strict for ``use_orig_params=False`` (hard error) and highly
    recommended for ``use_orig_params=True`` (user warning).
    """
    post_order_named_modules = _get_post_order_named_modules(root_module)
    visited_modules: Set[nn.Module] = set()
    for module_name, module in post_order_named_modules:
        if module in modules_to_wrap:
            param_to_fqn = _get_managed_param_to_fqn(module, ignored_params, visited_modules, module_name)
            frozen_param_fqns: List[str] = []
            frozen_param_numel = 0
            nonfrozen_param_fqns: List[str] = []
            nonfrozen_param_numel = 0
            for param, fqn in param_to_fqn.items():
                if param.requires_grad:
                    nonfrozen_param_fqns.append(fqn)
                    nonfrozen_param_numel += param.numel()
                else:
                    frozen_param_fqns.append(fqn)
                    frozen_param_numel += param.numel()
            if len(frozen_param_fqns) > 0 and len(nonfrozen_param_fqns) > 0:
                msg = f'{module_name} has both parameters with requires_grad=True and False.'
                if use_orig_params:
                    total_param_numel = frozen_param_numel + nonfrozen_param_numel
                    msg += f' We do not recommend wrapping such modules since the gradient memory usage will be higher than expected ({total_param_numel} numel instead of {nonfrozen_param_numel} numel before sharding via reduce-scatter). '
                else:
                    msg += ' FSDP does not support wrapping such modules when use_orig_params=False. '
                msg += 'If possible, wrap the frozen parameters with FSDP separately.\n'
                msg += f'The following parameters have requires_grad=True:\n{nonfrozen_param_fqns}\nThe following parameters have requires_grad=False:\n{frozen_param_fqns}'
                if use_orig_params:
                    warnings.warn(msg)
                else:
                    raise ValueError(msg)