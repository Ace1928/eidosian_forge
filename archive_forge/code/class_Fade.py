import math
import warnings
from typing import Callable, Optional, Sequence, Tuple, Union
import torch
from torch import Tensor
from torch.nn.modules.lazy import LazyModuleMixin
from torch.nn.parameter import UninitializedParameter
from torchaudio import functional as F
from torchaudio.functional.functional import (
class Fade(torch.nn.Module):
    """Add a fade in and/or fade out to an waveform.

    .. devices:: CPU CUDA

    .. properties:: Autograd TorchScript

    Args:
        fade_in_len (int, optional): Length of fade-in (time frames). (Default: ``0``)
        fade_out_len (int, optional): Length of fade-out (time frames). (Default: ``0``)
        fade_shape (str, optional): Shape of fade. Must be one of: "quarter_sine",
            ``"half_sine"``, ``"linear"``, ``"logarithmic"``, ``"exponential"``.
            (Default: ``"linear"``)

    Example
        >>> waveform, sample_rate = torchaudio.load("test.wav", normalize=True)
        >>> transform = transforms.Fade(fade_in_len=sample_rate, fade_out_len=2 * sample_rate, fade_shape="linear")
        >>> faded_waveform = transform(waveform)
    """

    def __init__(self, fade_in_len: int=0, fade_out_len: int=0, fade_shape: str='linear') -> None:
        super(Fade, self).__init__()
        self.fade_in_len = fade_in_len
        self.fade_out_len = fade_out_len
        self.fade_shape = fade_shape

    def forward(self, waveform: Tensor) -> Tensor:
        """
        Args:
            waveform (Tensor): Tensor of audio of dimension `(..., time)`.

        Returns:
            Tensor: Tensor of audio of dimension `(..., time)`.
        """
        waveform_length = waveform.size()[-1]
        device = waveform.device
        return self._fade_in(waveform_length, device) * self._fade_out(waveform_length, device) * waveform

    def _fade_in(self, waveform_length: int, device: torch.device) -> Tensor:
        fade = torch.linspace(0, 1, self.fade_in_len, device=device)
        ones = torch.ones(waveform_length - self.fade_in_len, device=device)
        if self.fade_shape == 'linear':
            fade = fade
        if self.fade_shape == 'exponential':
            fade = torch.pow(2, fade - 1) * fade
        if self.fade_shape == 'logarithmic':
            fade = torch.log10(0.1 + fade) + 1
        if self.fade_shape == 'quarter_sine':
            fade = torch.sin(fade * math.pi / 2)
        if self.fade_shape == 'half_sine':
            fade = torch.sin(fade * math.pi - math.pi / 2) / 2 + 0.5
        return torch.cat((fade, ones)).clamp_(0, 1)

    def _fade_out(self, waveform_length: int, device: torch.device) -> Tensor:
        fade = torch.linspace(0, 1, self.fade_out_len, device=device)
        ones = torch.ones(waveform_length - self.fade_out_len, device=device)
        if self.fade_shape == 'linear':
            fade = -fade + 1
        if self.fade_shape == 'exponential':
            fade = torch.pow(2, -fade) * (1 - fade)
        if self.fade_shape == 'logarithmic':
            fade = torch.log10(1.1 - fade) + 1
        if self.fade_shape == 'quarter_sine':
            fade = torch.sin(fade * math.pi / 2 + math.pi / 2)
        if self.fade_shape == 'half_sine':
            fade = torch.sin(fade * math.pi + math.pi / 2) / 2 + 0.5
        return torch.cat((ones, fade)).clamp_(0, 1)