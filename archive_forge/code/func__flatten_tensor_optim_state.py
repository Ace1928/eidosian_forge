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
def _flatten_tensor_optim_state(state_name: str, pos_dim_tensors: List[torch.Tensor], unflat_param_names: List[str], unflat_param_shapes: Sequence[torch.Size], handle: FlatParamHandle) -> torch.Tensor:
    """
    Flattens the positive-dimension tensor optimizer state given by the values
    ``tensors`` for the state ``state_name`` for a single flat parameter
    from ``handle`` corresponding to the unflattened parameter names
    ``unflat_param_names`` and unflatted parameter shapes
    ``unflat_param_shapes``. This flattens each unflattened parameter's tensor
    state into one tensor.

    NOTE: We use zero tensors for any unflattened parameters without state
    since some value is required to fill those entries. This assumes that the
    zero tensor is mathematically equivalent to having no state, which is true
    for Adam's "exp_avg" and "exp_avg_sq" but may not be true for all
    optimizers.

    Args:
        state_name (str): Optimizer state name.
        pos_dim_tensors (List[torch.Tensor]): Positive-dimension tensor
            optimizer state values for the unflattened parameters corresponding
            to the single flat parameter.
        unflat_param_names (List[str]): A :class:`list` of unflattened
            parameter names corresponding to the single flat parameter.
        unflat_param_shapes (List[torch.Size]): Unflattened parameter shapes
            corresponding to the single flat parameter.
        handle (FlatParamHandle): The flat parameter's handle.

    Returns:
        torch.Tensor: A flat tensor containing the optimizer state
        corresponding to ``state_name`` constructed by concatenating the
        unflattened parameter tensor states in ``pos_dim_tensors`` (using zero
        tensors for any unflattened parameters without the state).
    """
    flat_param = handle.flat_param
    non_none_tensors = [t for t in pos_dim_tensors if t is not None]
    dtypes = {t.dtype for t in non_none_tensors}
    if len(dtypes) != 1:
        raise ValueError(f'All unflattened parameters comprising a single flat parameter must have positive-dimension tensor state with the same dtype but got dtypes {dtypes} for state {state_name} and unflattened parameter names {unflat_param_names}')
    dtype = next(iter(dtypes))
    for tensor, shape in zip(pos_dim_tensors, unflat_param_shapes):
        if tensor is None and len(shape) == 0:
            raise ValueError('Flattening a zero-dimension parameter is not supported')
        elif tensor is not None and tensor.shape != shape:
            raise ValueError(f'Tensor optimizer state does not have same shape as its parameter: {tensor.shape} {shape}')
    cpu_device = torch.device('cpu')
    tensors_to_flatten = [torch.flatten(state_value.to(cpu_device)) if state_value is not None else torch.flatten(torch.zeros(size=shape, dtype=dtype, device=cpu_device)) for state_value, shape in zip(pos_dim_tensors, unflat_param_shapes)]
    flat_tensor = handle.flatten_tensors(tensors_to_flatten, handle._aligned_numel)
    flat_param_shape = flat_param._unpadded_unsharded_size
    assert flat_tensor.shape == flat_param_shape, f'tensor optim state: {flat_tensor.shape} flat parameter: {flat_param_shape}'
    return flat_tensor