from dataclasses import dataclass
from enum import auto, Enum
from typing import Optional, Sequence, Type
import torch
from torch.nn.modules.batchnorm import _BatchNorm
@dataclass
class CPUOffload:
    """
    This configures CPU offloading.

    Attributes:
        offload_params (bool): This specifies whether to offload parameters to
            CPU when not involved in computation. If ``True``, then this
            offloads gradients to CPU as well, meaning that the optimizer step
            runs on CPU.
    """
    offload_params: bool = False