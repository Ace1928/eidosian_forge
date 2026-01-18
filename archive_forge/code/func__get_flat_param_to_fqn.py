import copy
import functools
import logging
import warnings
from contextlib import ExitStack
from dataclasses import dataclass, field
from typing import (
import torch
import torch.distributed as dist
import torch.distributed.fsdp._traversal_utils as traversal_utils
import torch.nn as nn
from torch.distributed._shard.sharded_tensor import ShardedTensor
from torch.distributed._tensor import DTensor, Replicate
from torch.distributed.checkpoint._state_dict_utils import _gather_state_dict
from torch.distributed.distributed_c10d import _get_pg_default_device
from torch.distributed.fsdp._common_utils import (
from torch.distributed.fsdp._debug_utils import SimpleProfiler
from torch.distributed.fsdp._flat_param import FlatParameter, FlatParamHandle
from torch.distributed.fsdp._fsdp_extensions import (
from torch.distributed.fsdp._runtime_utils import (
from torch.distributed.fsdp.api import (
from torch.utils._pytree import tree_map_only
def _get_flat_param_to_fqn(model: torch.nn.Module) -> Dict[FlatParameter, str]:
    """
    Constructs a mapping from ``FlatParameter`` to a cleaned (devoid of prefixes
    from wrappers) fully qualified name (FQN). Note that this FQN is "non-canonical"
    because ``FlatParameter``  s do not come from the original module but are
    registered only after FSDP has been applied. This function returns the FSDP-given
    name for the ``FlatParameter`` (usually module._flat_param) as opposed to the
    canonical FQNs returned for ``FlatParameter`` s in ``_common_utils._get_param_to_fqns(...)``).

    Consequently, this function will only return a non-empty mapping if FSDP was
    applied with ``use_orig_params=False`` as, otherwise, the original parameters
    are used within the module and there would be no ``FlatParameter`` s in the module.

    """

    def module_fn(module, prefix, tree_level, flat_param_to_fqn):
        for param_name, param in _named_parameters_with_duplicates(module, recurse=False):
            if not isinstance(param, FlatParameter):
                continue
            fqn = clean_tensor_name(prefix + param_name)
            flat_param_to_fqn[param] = fqn

    def return_fn(flat_param_to_fqn):
        return flat_param_to_fqn
    flat_param_to_fqn_ret: Dict[FlatParameter, str] = {}
    return _apply_to_modules(model, module_fn, return_fn, [fqn for fqn, _ in _named_parameters_with_duplicates(model)], flat_param_to_fqn_ret)