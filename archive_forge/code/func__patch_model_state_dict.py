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
def _patch_model_state_dict(model: nn.Module, *, options: Optional[StateDictOptions]=None) -> None:
    """Patch the ``state_dict`` and ``load_state_dict`` attributes of ``model``.

    Patch the ``state_dict`` and ``load_state_dict`` attributes of ``model`` to
    be a partial function to call ``get_state_dict`` and ``set_state_dict``.

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
    _state_dict_call = functools.partial(get_model_state_dict, model=model, options=options)

    def state_dict_call():
        return _state_dict_call()
    model.state_dict = state_dict_call
    _load_state_dict_call = functools.partial(set_model_state_dict, model=model, options=options)

    def load_state_dict_call(state_dict: Dict[str, Any]):
        _load_state_dict_call(model_state_dict=state_dict)
    model.load_state_dict = load_state_dict_call
    _patched_state_dict.add(state_dict_call)
    _patched_state_dict.add(load_state_dict_call)