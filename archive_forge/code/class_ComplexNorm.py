from typing import Optional
import torch
import torchaudio
from torch import Tensor
import torch.nn as nn
class ComplexNorm(nn.Module):
    """Compute the norm of complex tensor input.

    Extension of `torchaudio.functional.complex_norm` with mono

    Args:
        mono (bool): Downmix to single channel after applying power norm
            to maximize
    """

    def __init__(self, mono: bool=False):
        super(ComplexNorm, self).__init__()
        self.mono = mono

    def forward(self, spec: Tensor) -> Tensor:
        """
        Args:
            spec: complex_tensor (Tensor): Tensor shape of
                `(..., complex=2)`

        Returns:
            Tensor: Power/Mag of input
                `(...,)`
        """
        spec = torch.abs(torch.view_as_complex(spec))
        if self.mono:
            spec = torch.mean(spec, 1, keepdim=True)
        return spec