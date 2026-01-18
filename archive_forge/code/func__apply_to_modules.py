import logging
import traceback
import warnings
import weakref
from enum import auto, Enum
from functools import partial
from typing import (
import torch
import torch.distributed as dist
import torch.distributed.fsdp._flat_param as flat_param_file
import torch.nn as nn
from torch.distributed._composable_state import _get_module_state, _State
from torch.distributed._tensor.device_mesh import DeviceMesh
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
from torch.distributed.fsdp._fsdp_extensions import FSDPExtensions
from torch.distributed.utils import _apply_to_tensors
from torch.utils._mode_utils import no_dispatch
from .api import (
def _apply_to_modules(root_module: torch.nn.Module, module_fn: Callable, return_fn: Callable, filter_fqns: Optional[List[str]]=None, *args, **kwargs):
    """
    Performs a pre-order traversal of the modules in the hierarchy rooted at
    ``root_module``, applying ``module_fn`` at each module and finally
    returning a value using ``return_fn``. The traversal constructs the full
    module prefix name (e.g. "module.submodule." just like in model state dict)
    and makes that available to ``module_fn``.

    ``filter_fqns`` is used because some module may have its own prefix similar
    to ``FullyShardedDataParallel`` and the ``named_parameters()`` is overwritten
    to remove the prefix.
    """

    def f(module: torch.nn.Module, prefix: str, tree_level: int, *args, **kwargs):
        module_fn(module, prefix, tree_level, *args, **kwargs)
        for submodule_name, submodule in module.named_children():
            if submodule is None:
                continue
            new_prefix = prefix + submodule_name + '.'
            new_tree_level = tree_level + 1
            if filter_fqns is not None:
                for fqn in filter_fqns:
                    if fqn.startswith(new_prefix):
                        break
                else:
                    if submodule_name == '_fsdp_wrapped_module' or submodule_name == '_dmp_wrapped_module':
                        if not torch.distributed._functional_collectives.is_torchdynamo_compiling():
                            warnings.warn(f'An unexpected prefix is detected. This case  should only happen when using DMP with FSDP. prefix = {prefix}, submodule_name = {submodule_name}')
                        new_prefix = prefix
                    elif submodule_name == 'module':
                        warnings.warn(f'An unexpected prefix is detected. This case  should only happen when DDP wraps the outer  modules while FSDP wraps the inner ones.prefix = {prefix}, submodule_name = {submodule_name}')
                        new_prefix = prefix
            f(submodule, new_prefix, new_tree_level, *args, **kwargs)
    f(root_module, '', 0, *args, **kwargs)
    return return_fn(*args, **kwargs)