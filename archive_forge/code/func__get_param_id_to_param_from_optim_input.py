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
def _get_param_id_to_param_from_optim_input(model: nn.Module, optim_input: Optional[Union[List[Dict[str, Any]], Iterable[nn.Parameter]]]=None) -> Dict[int, nn.Parameter]:
    """
    Constructs a mapping from parameter IDs to parameters. This may be used
    both for models with ``FlatParameter`` s and without.

    NOTE: This method is only preserved for backward compatibility. The method
    :meth:`_get_param_key_to_param` is the preferred code path that does not
    rely on ``optim_input``.

    NOTE: We critically assume that, whether the optimizer input is a list of
    parameters or a list of parameter groups, :class:`torch.optim.Optimizer`
    enumerates the parameter IDs in order. In other words, for a parameter list
    input, the parameter IDs should be in that list order, and for a parameter
    groups input, the parameter IDs should be in order within each parameter
    group and in order across parameter groups.

    Args:
        model (nn.Module): Model whose parameters are passed into the
            optimizer.
        optim_input (Optional[Union[List[Dict[str, Any]],
        Iterable[nn.Parameter]]]): Input passed into the optimizer
            representing either a :class:`list` of parameter groups or an
            iterable of parameters; if ``None``, then this method assumes the
            input was ``model.parameters()``. (Default: ``None``)

    Returns:
        List[nn.Parameter]: Mapping from parameter IDs to parameters,
        where the parameter ID is implicitly the index in the :class:`list`.
    """
    if optim_input is None:
        return dict(enumerate(model.parameters()))
    try:
        params = cast(List[nn.Parameter], list(optim_input))
    except TypeError as e:
        raise TypeError(f'Optimizer input should be an iterable of Tensors or dicts, but got {optim_input}') from e
    if len(params) == 0:
        raise ValueError('Optimizer input should not be empty')
    all_tensors = True
    all_dicts = True
    for param in params:
        all_tensors &= isinstance(param, torch.Tensor)
        all_dicts &= isinstance(param, dict)
    if not all_tensors and (not all_dicts):
        raise TypeError('Optimizer input should be an iterable of Tensors or dicts')
    if all_tensors:
        return dict(enumerate(params))
    assert all_dicts
    param_id_to_param: List[nn.Parameter] = []
    for param_group in params:
        has_params_key = 'params' in param_group
        assert has_params_key, 'A parameter group should map "params" to a list of the parameters in the group'
        for param in param_group['params']:
            param_id_to_param.append(param)
    return dict(enumerate(param_id_to_param))