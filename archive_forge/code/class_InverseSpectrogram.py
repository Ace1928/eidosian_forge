import math
import warnings
from typing import Callable, Optional, Sequence, Tuple, Union
import torch
from torch import Tensor
from torch.nn.modules.lazy import LazyModuleMixin
from torch.nn.parameter import UninitializedParameter
from torchaudio import functional as F
from torchaudio.functional.functional import (
class InverseSpectrogram(torch.nn.Module):
    """Create an inverse spectrogram to recover an audio signal from a spectrogram.

    .. devices:: CPU CUDA

    .. properties:: Autograd TorchScript

    Args:
        n_fft (int, optional): Size of FFT, creates ``n_fft // 2 + 1`` bins. (Default: ``400``)
        win_length (int or None, optional): Window size. (Default: ``n_fft``)
        hop_length (int or None, optional): Length of hop between STFT windows. (Default: ``win_length // 2``)
        pad (int, optional): Two sided padding of signal. (Default: ``0``)
        window_fn (Callable[..., Tensor], optional): A function to create a window tensor
            that is applied/multiplied to each frame/window. (Default: ``torch.hann_window``)
        normalized (bool or str, optional): Whether the stft output was normalized by magnitude. If input is str,
            choices are ``"window"`` and ``"frame_length"``, dependent on normalization mode. ``True`` maps to
            ``"window"``. (Default: ``False``)
        wkwargs (dict or None, optional): Arguments for window function. (Default: ``None``)
        center (bool, optional): whether the signal in spectrogram was padded on both sides so
            that the :math:`t`-th frame is centered at time :math:`t \\times \\text{hop\\_length}`.
            (Default: ``True``)
        pad_mode (string, optional): controls the padding method used when
            :attr:`center` is ``True``. (Default: ``"reflect"``)
        onesided (bool, optional): controls whether spectrogram was used to return half of results to
            avoid redundancy (Default: ``True``)

    Example
        >>> batch, freq, time = 2, 257, 100
        >>> length = 25344
        >>> spectrogram = torch.randn(batch, freq, time, dtype=torch.cdouble)
        >>> transform = transforms.InverseSpectrogram(n_fft=512)
        >>> waveform = transform(spectrogram, length)
    """
    __constants__ = ['n_fft', 'win_length', 'hop_length', 'pad', 'power', 'normalized']

    def __init__(self, n_fft: int=400, win_length: Optional[int]=None, hop_length: Optional[int]=None, pad: int=0, window_fn: Callable[..., Tensor]=torch.hann_window, normalized: Union[bool, str]=False, wkwargs: Optional[dict]=None, center: bool=True, pad_mode: str='reflect', onesided: bool=True) -> None:
        super(InverseSpectrogram, self).__init__()
        self.n_fft = n_fft
        self.win_length = win_length if win_length is not None else n_fft
        self.hop_length = hop_length if hop_length is not None else self.win_length // 2
        window = window_fn(self.win_length) if wkwargs is None else window_fn(self.win_length, **wkwargs)
        self.register_buffer('window', window)
        self.pad = pad
        self.normalized = normalized
        self.center = center
        self.pad_mode = pad_mode
        self.onesided = onesided

    def forward(self, spectrogram: Tensor, length: Optional[int]=None) -> Tensor:
        """
        Args:
            spectrogram (Tensor): Complex tensor of audio of dimension (..., freq, time).
            length (int or None, optional): The output length of the waveform.

        Returns:
            Tensor: Dimension (..., time), Least squares estimation of the original signal.
        """
        return F.inverse_spectrogram(spectrogram, length, self.pad, self.window, self.n_fft, self.hop_length, self.win_length, self.normalized, self.center, self.pad_mode, self.onesided)