from dataclasses import dataclass
from enum import auto, Enum
from typing import Optional, Sequence, Type
import torch
from torch.nn.modules.batchnorm import _BatchNorm
@dataclass
class OptimStateDictConfig:
    """
    ``OptimStateDictConfig`` is the base class for all ``optim_state_dict``
    configuration classes.  Users should instantiate a child class (e.g.
    ``FullOptimStateDictConfig``) in order to configure settings for the
    corresponding ``optim_state_dict`` type supported by FSDP.

    Attributes:
        offload_to_cpu (bool): If ``True``, then FSDP offloads the state dict's
            tensor values to CPU, and if ``False``, then FSDP keeps them on the
            original device (which is GPU unless parameter CPU offloading is
            enabled). (Default: ``True``)
    """
    offload_to_cpu: bool = True