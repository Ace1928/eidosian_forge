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
def _get_param_key_to_param(optim: torch.optim.Optimizer, model: Optional[nn.Module]=None, is_named_optimizer: bool=False, param_to_fqns: Optional[Dict[nn.Parameter, List[str]]]=None, flat_param_to_fqn: Optional[Dict[FlatParameter, str]]=None) -> Dict[Union[int, str], nn.Parameter]:
    """
    Constructs a mapping from parameter keys to parameters. For the regular
    optimizers, the keys are parameter IDs. For NamedOptimizer, the keys
    are FQNs. This API may be used both for models with ``FlatParameter`` s and
    without.
    """
    clean_fqn_to_curr_fqn: Dict[str, str] = {}
    if is_named_optimizer:
        assert param_to_fqns is not None and flat_param_to_fqn is not None, 'The optimizer is a NamedOptimizer, `param_to_fqns` must not be None.'
        assert model is not None
        for key, _ in _named_parameters_with_duplicates(model):
            clean_fqn_to_curr_fqn[clean_tensor_name(key)] = key
    param_key_to_param: Dict[Union[str, int], nn.Parameter] = {}
    pid = 0
    for param_group in optim.param_groups:
        if is_named_optimizer:
            for param in param_group['params']:
                assert flat_param_to_fqn is not None
                if param in flat_param_to_fqn:
                    key = flat_param_to_fqn[param]
                else:
                    assert param_to_fqns is not None
                    assert len(param_to_fqns[param]) == 1
                    key = param_to_fqns[param][0]
                try:
                    key = clean_fqn_to_curr_fqn[key]
                except KeyError as e:
                    raise KeyError(f"Can't find {key} from {list(clean_fqn_to_curr_fqn.keys())}.") from e
                param_key_to_param[key] = param
        else:
            for param in param_group['params']:
                param_key_to_param[pid] = param
                pid += 1
    return param_key_to_param