import math
import warnings
from typing import Callable, Optional, Sequence, Tuple, Union
import torch
from torch import Tensor
from torch.nn.modules.lazy import LazyModuleMixin
from torch.nn.parameter import UninitializedParameter
from torchaudio import functional as F
from torchaudio.functional.functional import (
class InverseMelScale(torch.nn.Module):
    """Estimate a STFT in normal frequency domain from mel frequency domain.

    .. devices:: CPU CUDA

    It minimizes the euclidian norm between the input mel-spectrogram and the product between
    the estimated spectrogram and the filter banks using `torch.linalg.lstsq`.

    Args:
        n_stft (int): Number of bins in STFT. See ``n_fft`` in :class:`Spectrogram`.
        n_mels (int, optional): Number of mel filterbanks. (Default: ``128``)
        sample_rate (int, optional): Sample rate of audio signal. (Default: ``16000``)
        f_min (float, optional): Minimum frequency. (Default: ``0.``)
        f_max (float or None, optional): Maximum frequency. (Default: ``sample_rate // 2``)
        norm (str or None, optional): If "slaney", divide the triangular mel weights by the width of the mel band
            (area normalization). (Default: ``None``)
        mel_scale (str, optional): Scale to use: ``htk`` or ``slaney``. (Default: ``htk``)
        driver (str, optional): Name of the LAPACK/MAGMA method to be used for `torch.lstsq`.
            For CPU inputs the valid values are ``"gels"``, ``"gelsy"``, ``"gelsd"``, ``"gelss"``.
            For CUDA input, the only valid driver is ``"gels"``, which assumes that A is full-rank.
            (Default: ``"gels``)

    Example
        >>> waveform, sample_rate = torchaudio.load("test.wav", normalize=True)
        >>> mel_spectrogram_transform = transforms.MelSpectrogram(sample_rate, n_fft=1024)
        >>> mel_spectrogram = mel_spectrogram_transform(waveform)
        >>> inverse_melscale_transform = transforms.InverseMelScale(n_stft=1024 // 2 + 1)
        >>> spectrogram = inverse_melscale_transform(mel_spectrogram)
    """
    __constants__ = ['n_stft', 'n_mels', 'sample_rate', 'f_min', 'f_max']

    def __init__(self, n_stft: int, n_mels: int=128, sample_rate: int=16000, f_min: float=0.0, f_max: Optional[float]=None, norm: Optional[str]=None, mel_scale: str='htk', driver: str='gels') -> None:
        super(InverseMelScale, self).__init__()
        self.n_mels = n_mels
        self.sample_rate = sample_rate
        self.f_max = f_max or float(sample_rate // 2)
        self.f_min = f_min
        self.driver = driver
        if f_min > self.f_max:
            raise ValueError('Require f_min: {} <= f_max: {}'.format(f_min, self.f_max))
        if driver not in ['gels', 'gelsy', 'gelsd', 'gelss']:
            raise ValueError(f'driver must be one of ["gels", "gelsy", "gelsd", "gelss"]. Found {driver}.')
        fb = F.melscale_fbanks(n_stft, self.f_min, self.f_max, self.n_mels, self.sample_rate, norm, mel_scale)
        self.register_buffer('fb', fb)

    def forward(self, melspec: Tensor) -> Tensor:
        """
        Args:
            melspec (Tensor): A Mel frequency spectrogram of dimension (..., ``n_mels``, time)

        Returns:
            Tensor: Linear scale spectrogram of size (..., freq, time)
        """
        shape = melspec.size()
        melspec = melspec.view(-1, shape[-2], shape[-1])
        n_mels, time = (shape[-2], shape[-1])
        freq, _ = self.fb.size()
        if self.n_mels != n_mels:
            raise ValueError('Expected an input with {} mel bins. Found: {}'.format(self.n_mels, n_mels))
        specgram = torch.relu(torch.linalg.lstsq(self.fb.transpose(-1, -2)[None], melspec, driver=self.driver).solution)
        specgram = specgram.view(shape[:-2] + (freq, time))
        return specgram