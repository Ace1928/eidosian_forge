from dataclasses import dataclass
from enum import auto, Enum
from typing import Optional, Sequence, Type
import torch
from torch.nn.modules.batchnorm import _BatchNorm
@dataclass
class StateDictConfig:
    """
    ``StateDictConfig`` is the base class for all ``state_dict`` configuration
    classes. Users should instantiate a child class (e.g.
    ``FullStateDictConfig``) in order to configure settings for the
    corresponding ``state_dict`` type supported by FSDP.

    Attributes:
        offload_to_cpu (bool): If ``True``, then FSDP offloads the state dict
            values to CPU, and if ``False``, then FSDP keeps them on GPU.
            (Default: ``False``)
    """
    offload_to_cpu: bool = False