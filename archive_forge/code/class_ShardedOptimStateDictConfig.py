from dataclasses import dataclass
from enum import auto, Enum
from typing import Optional, Sequence, Type
import torch
from torch.nn.modules.batchnorm import _BatchNorm
@dataclass
class ShardedOptimStateDictConfig(OptimStateDictConfig):
    """
    ``ShardedOptimStateDictConfig`` is a config class meant to be used with
    ``StateDictType.SHARDED_STATE_DICT``.

    Attributes:
        _use_dtensor (bool): If ``True``, then FSDP saves the state dict values
            as ``DTensor``, and if ``False``, then FSDP saves them as
            ``ShardedTensor``. (Default: ``False``)

    .. warning:: ``_use_dtensor`` is a private field of :class:`ShardedOptimStateDictConfig`
      and it is used by FSDP to determine the type of state dict values. Users should not
      manually modify ``_use_dtensor``.
    """
    _use_dtensor: bool = False