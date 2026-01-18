import math
import warnings
from typing import Callable, Optional, Sequence, Tuple, Union
import torch
from torch import Tensor
from torch.nn.modules.lazy import LazyModuleMixin
from torch.nn.parameter import UninitializedParameter
from torchaudio import functional as F
from torchaudio.functional.functional import (
class Loudness(torch.nn.Module):
    """Measure audio loudness according to the ITU-R BS.1770-4 recommendation.

    .. devices:: CPU CUDA

    .. properties:: TorchScript

    Args:
        sample_rate (int): Sample rate of audio signal.

    Example
        >>> waveform, sample_rate = torchaudio.load("test.wav", normalize=True)
        >>> transform = transforms.Loudness(sample_rate)
        >>> loudness = transform(waveform)

    Reference:
        - https://www.itu.int/rec/R-REC-BS.1770-4-201510-I/en
    """
    __constants__ = ['sample_rate']

    def __init__(self, sample_rate: int):
        super(Loudness, self).__init__()
        self.sample_rate = sample_rate

    def forward(self, wavefrom: Tensor):
        """
        Args:
            waveform(torch.Tensor): audio waveform of dimension `(..., channels, time)`

        Returns:
            Tensor: loudness estimates (LKFS)
        """
        return F.loudness(wavefrom, self.sample_rate)