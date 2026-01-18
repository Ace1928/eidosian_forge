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
def _unflatten_param_groups(state_dict: Dict[str, Any], param_key_to_param: Dict[Union[int, str], nn.Parameter], param_to_fqns: Dict[nn.Parameter, List[str]]) -> List[Dict[str, Any]]:
    param_groups: List[Dict[str, Any]] = []
    for flat_param_group in state_dict['param_groups']:
        unflat_param_group = copy.deepcopy(flat_param_group)
        param_group_params = [param_key_to_param[flat_param_key] for flat_param_key in flat_param_group['params']]
        nested_unflat_param_names = [param_to_fqns[param] for param in param_group_params]
        unflat_param_group['params'] = [unflat_param_name for unflat_param_names in nested_unflat_param_names for unflat_param_name in unflat_param_names]
        param_groups.append(unflat_param_group)
    return param_groups