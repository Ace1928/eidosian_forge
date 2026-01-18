import collections
import functools
import inspect
import warnings
from functools import partial
from typing import Any, Callable, Dict, List, Set, Tuple, Type, Union
import torch.nn as nn
from torch.distributed.fsdp._common_utils import (
from torch.distributed.fsdp.wrap import (
def _get_managed_param_to_fqn(module_to_wrap: nn.Module, ignored_params: Set[nn.Parameter], visited_modules: Set[nn.Module], root_prefix: str) -> Dict[nn.Parameter, str]:
    """
    This returns a dict that maps managed parameter to its FQN for the given
    ``module_to_wrap``. The dict's keys are exactly the parameters that would
    be managed by the module, where this is achieved by calling this function
    on the modules to wrap in reverse topological order, destructively updating
    ``visited_modules``, and not traversing into those modules. The FQNs are
    prefixed from the root (via ``root_prefix``) to be more informative.

    NOTE: This function is meant to be called pre-wrapping and iteratively in
    reverse topological order to cover the full module tree. This differs from
    the ``_get_param_to_fqn()`` function meant to be called post-wrapping and
    on the full module tree in one shot. Given those differences, we do not try
    to unify the two.
    """
    param_to_fqn: Dict[nn.Parameter, str] = {}
    queue = collections.deque([(module_to_wrap, root_prefix)])
    visited_modules.add(module_to_wrap)
    while queue:
        module, prefix = queue.popleft()
        for param_name, param in module.named_parameters(recurse=False):
            if param not in ignored_params:
                fqn = param_name if prefix == '' else prefix + '.' + param_name
                param_to_fqn[param] = fqn
        for child_module_name, child_module in module.named_children():
            if child_module is None:
                continue
            if child_module not in visited_modules:
                visited_modules.add(child_module)
                child_prefix = child_module_name if prefix == '' else prefix + '.' + child_module_name
                queue.append((child_module, child_prefix))
    return param_to_fqn