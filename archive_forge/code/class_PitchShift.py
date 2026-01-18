import math
import warnings
from typing import Callable, Optional, Sequence, Tuple, Union
import torch
from torch import Tensor
from torch.nn.modules.lazy import LazyModuleMixin
from torch.nn.parameter import UninitializedParameter
from torchaudio import functional as F
from torchaudio.functional.functional import (
class PitchShift(LazyModuleMixin, torch.nn.Module):
    """Shift the pitch of a waveform by ``n_steps`` steps.

    .. devices:: CPU CUDA

    .. properties:: TorchScript

    Args:
        waveform (Tensor): The input waveform of shape `(..., time)`.
        sample_rate (int): Sample rate of `waveform`.
        n_steps (int): The (fractional) steps to shift `waveform`.
        bins_per_octave (int, optional): The number of steps per octave (Default : ``12``).
        n_fft (int, optional): Size of FFT, creates ``n_fft // 2 + 1`` bins (Default: ``512``).
        win_length (int or None, optional): Window size. If None, then ``n_fft`` is used. (Default: ``None``).
        hop_length (int or None, optional): Length of hop between STFT windows. If None, then ``win_length // 4``
            is used (Default: ``None``).
        window (Tensor or None, optional): Window tensor that is applied/multiplied to each frame/window.
            If None, then ``torch.hann_window(win_length)`` is used (Default: ``None``).

    Example
        >>> waveform, sample_rate = torchaudio.load("test.wav", normalize=True)
        >>> transform = transforms.PitchShift(sample_rate, 4)
        >>> waveform_shift = transform(waveform)  # (channel, time)
    """
    __constants__ = ['sample_rate', 'n_steps', 'bins_per_octave', 'n_fft', 'win_length', 'hop_length']
    kernel: UninitializedParameter
    width: int

    def __init__(self, sample_rate: int, n_steps: int, bins_per_octave: int=12, n_fft: int=512, win_length: Optional[int]=None, hop_length: Optional[int]=None, window_fn: Callable[..., Tensor]=torch.hann_window, wkwargs: Optional[dict]=None) -> None:
        super().__init__()
        self.n_steps = n_steps
        self.bins_per_octave = bins_per_octave
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.win_length = win_length if win_length is not None else n_fft
        self.hop_length = hop_length if hop_length is not None else self.win_length // 4
        window = window_fn(self.win_length) if wkwargs is None else window_fn(self.win_length, **wkwargs)
        self.register_buffer('window', window)
        rate = 2.0 ** (-float(n_steps) / bins_per_octave)
        self.orig_freq = int(sample_rate / rate)
        self.gcd = math.gcd(int(self.orig_freq), int(sample_rate))
        if self.orig_freq != sample_rate:
            self.width = -1
            self.kernel = UninitializedParameter(device=None, dtype=None)

    def initialize_parameters(self, input):
        if self.has_uninitialized_params():
            if self.orig_freq != self.sample_rate:
                with torch.no_grad():
                    kernel, self.width = _get_sinc_resample_kernel(self.orig_freq, self.sample_rate, self.gcd, dtype=input.dtype, device=input.device)
                    self.kernel.materialize(kernel.shape)
                    self.kernel.copy_(kernel)

    def forward(self, waveform: Tensor) -> Tensor:
        """
        Args:
            waveform (Tensor): Tensor of audio of dimension `(..., time)`.

        Returns:
            Tensor: The pitch-shifted audio of shape `(..., time)`.
        """
        shape = waveform.size()
        waveform_stretch = _stretch_waveform(waveform, self.n_steps, self.bins_per_octave, self.n_fft, self.win_length, self.hop_length, self.window)
        if self.orig_freq != self.sample_rate:
            waveform_shift = _apply_sinc_resample_kernel(waveform_stretch, self.orig_freq, self.sample_rate, self.gcd, self.kernel, self.width)
        else:
            waveform_shift = waveform_stretch
        return _fix_waveform_shape(waveform_shift, shape)