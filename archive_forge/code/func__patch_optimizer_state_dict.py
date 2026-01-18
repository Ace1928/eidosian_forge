import contextlib
import functools
import gc
from dataclasses import asdict, dataclass, field
from itertools import chain
from typing import (
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed._shard.sharded_tensor import ShardedTensor
from torch.distributed._tensor import DTensor
from torch.distributed.checkpoint._state_dict_utils import (
from torch.distributed.fsdp import (
from torch.distributed.fsdp._common_utils import (
from torch.nn.modules.module import _IncompatibleKeys
from torch.nn.parallel import DistributedDataParallel as DDP
@no_type_check
def _patch_optimizer_state_dict(model: nn.Module, *, optimizers: Tuple[torch.optim.Optimizer, ...], options: Optional[StateDictOptions]=None) -> None:
    """Patch the ``state_dict`` and ``load_state_dict`` attributes of ``optimizers``.

    Patch the ``state_dict`` and ``load_state_dict`` attributes of ``optimizers`` to
    be a partial function to call ``get_state_dict`` and ``set_state_dict``.

    Note that if there are multiple optimizers, all of the optimizers will be patched.
    So users only need to call one of the state_dict() to get the full result.

    Example:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from torch.distributed.checkpoint.state_dict import patch_model_state_dict

        model = fsdp(model)
        patch_model_state_dict(model)

    Args:
        model (nn.Module): the nn.Module to the model.
        options (StateDictOptions): the options to control how
            model state_dict and optimizer state_dict should be loaded. See
            `StateDictOptions` for the details.
    Returns:
        None
    """
    _state_dict_call = functools.partial(get_optimizer_state_dict, model=model, optimizers=optimizers, options=options)

    def state_dict_call():
        return _state_dict_call()
    _load_state_dict_call = functools.partial(set_optimizer_state_dict, model=model, optimizers=optimizers, options=options)

    def load_state_dict_call(state_dict: Dict[str, Any]):
        _load_state_dict_call(optim_state_dict=state_dict)
    _patched_state_dict.add(state_dict_call)
    _patched_state_dict.add(load_state_dict_call)
    optimizers = (optimizers,) if isinstance(optimizers, torch.optim.Optimizer) else tuple(optimizers)
    for optim in optimizers:
        optim.state_dict = state_dict_call
        optim.load_state_dict = load_state_dict_call