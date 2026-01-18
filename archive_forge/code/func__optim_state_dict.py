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
@torch.no_grad()
def _optim_state_dict(model: nn.Module, optim: torch.optim.Optimizer, optim_state_dict: Dict[str, Any], optim_input: Optional[Union[List[Dict[str, Any]], Iterable[nn.Parameter]]], rank0_only: bool, shard_state: bool, group: Optional[dist.ProcessGroup], using_optim_input: bool, use_orig_params: bool=False, cpu_offload: bool=True) -> Dict[str, Any]:
    """
    Consolidates the optimizer state and returns it as a :class:`dict`
    following the convention of :meth:`torch.optim.Optimizer.state_dict`,
    i.e. with keys ``"state"`` and ``"param_groups"``.
    The flat parameters in ``FSDP`` modules contained in ``model`` are mapped
    back to their unflattened parameters.

    Parameter keys are not well-defined. For a regular optimizer, the optimizer
    state_dict contains a mapping from parameter IDs to parameter states.
    Parameter IDs are the order of parameters in ``optim.param_groups()`` across
    all the groups. This API also allows user to pass ``optim_input`` for the
    mapping between parameters and parameter IDs. Using ``optim_input`` is being
    deprecated.

    If the optimizer is a ``NamedOptimizer``, the optimizer state_dict does not
    contain parameter IDs mapping but a mapping from parameter FQNs to parameter
    states. This API finds the mapping from FQNs to parameters if the optimizer
    is a ``NamedOptimizer``.

    If ``use_orig_params`` is True, each rank will have all FSDP-managed
    parameters but some of these parameters may be empty due to the sharding.
    For a regular optim.Optimizer, states for those empty parameters will
    not be initialized. So, when aggregating the FQNs across ranks, no assert
    will be raised on a rank even if it does not have all the states -- it is
    valid and FSDP knows how to aggregate them. However, FSDP has to ignore
    handling those parameters that are not managed by FSDP and do not exist on
    the local rank -- those are managed by other parallelisms and FSDP does not
    know how to handle/aggregate them.

    Args:
        model (nn.Module): Root module (which may or may not be a
            :class:`FullyShardedDataParallel` instance) whose parameters
            were passed into the optimizer ``optim``.
        optim (torch.optim.Optimizer): Optimizer for ``model`` 's
            parameters.
        rank0_only (bool): If ``True``, saves the populated :class:`dict`
            only on rank 0; if ``False``, saves it on all ranks. (Default:
            ``True``)
        shard_state (bool): If ``True``, shard and distribute all
            non-zero-dimension states.

    Returns:
        Dict[str, Any]: A :class:`dict` containing the optimizer state for
        ``model`` 's original unflattened parameters and including keys
        "state" and "param_groups" following the convention of
        :meth:`torch.optim.Optimizer.state_dict`. If ``rank0_only=False``,
        then nonzero ranks return an empty :class:`dict`.
    """
    SimpleProfiler.reset()
    cm = ExitStack()
    cm.enter_context(SimpleProfiler.profile(SimpleProfiler.Type.ALL))
    _reset_flat_param_grad_info_if_needed(traversal_utils._get_fsdp_handles(model))
    to_save = not rank0_only or dist.get_rank(group) == 0 or shard_state
    with SimpleProfiler.profile('preprocessing'):
        param_to_fqns = _get_param_to_fqns(model)
        flat_param_to_fqn = _get_flat_param_to_fqn(model)
        is_named_optimizer = _is_named_optimizer(optim_state_dict)
        param_key_to_param = cast(Dict[Union[int, str], nn.Parameter], _get_param_id_to_param_from_optim_input(model, optim_input) if using_optim_input else _get_param_key_to_param(optim, model, is_named_optimizer, param_to_fqns, flat_param_to_fqn))
        fqn_to_fsdp_param_info = _get_fqn_to_fsdp_param_info(model)
    with SimpleProfiler.profile('preprocessing_with_comm'):
        all_optim_state_keys, optim_state_key_to_param_key = _map_param_key_to_optim_keys(optim_state_dict, group, param_key_to_param, param_to_fqns, fqn_to_fsdp_param_info, merge_keys=use_orig_params)
    with SimpleProfiler.profile('state_converting'):
        convert_fn = _convert_state_with_orig_params if use_orig_params else _convert_state_with_flat_params
        fsdp_osd_state = convert_fn(all_optim_state_keys, optim_state_key_to_param_key, fqn_to_fsdp_param_info, optim_state_dict['state'], to_save, shard_state, cpu_offload)
    if not to_save:
        return {}
    fsdp_osd: Dict[str, Any] = {'state': fsdp_osd_state}
    flat_param_fqns = set(flat_param_to_fqn.values())
    for key, value in optim_state_dict['state'].items():
        if key in fsdp_osd_state:
            continue
        if key in flat_param_fqns:
            continue
        if key in param_key_to_param:
            continue
        warnings.warn(f'Found a optim state, {key}, that FSDP cannot process. FSDP will directly copy everything to the returned state_dict. In most cases, this is a user-defined state that is not associated with any particular parameter. Another possible case is this state is managed by TorchRec. Otherwise, there may  be a mismatched assumption of optim_state_dict of this mode.')
        fsdp_osd_state[key] = value
    if 'param_groups' in optim_state_dict:
        fsdp_osd['param_groups'] = _unflatten_param_groups(optim_state_dict, param_key_to_param, param_to_fqns)
    cm.close()
    SimpleProfiler.dump_and_reset('FSDP _optim_state_dict() profiling: ')
    return fsdp_osd