from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv1d, ConvTranspose1d
@torch.jit.interface
class ResBlockInterface(torch.nn.Module):
    """Interface for ResBlock - necessary to make type annotations in ``HiFiGANVocoder.forward`` compatible
    with TorchScript
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass