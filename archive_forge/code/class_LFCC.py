import math
import warnings
from typing import Callable, Optional, Sequence, Tuple, Union
import torch
from torch import Tensor
from torch.nn.modules.lazy import LazyModuleMixin
from torch.nn.parameter import UninitializedParameter
from torchaudio import functional as F
from torchaudio.functional.functional import (
class LFCC(torch.nn.Module):
    """Create the linear-frequency cepstrum coefficients from an audio signal.

    .. devices:: CPU CUDA

    .. properties:: Autograd TorchScript

    By default, this calculates the LFCC on the DB-scaled linear filtered spectrogram.
    This is not the textbook implementation, but is implemented here to
    give consistency with librosa.

    This output depends on the maximum value in the input spectrogram, and so
    may return different values for an audio clip split into snippets vs. a
    a full clip.

    Args:
        sample_rate (int, optional): Sample rate of audio signal. (Default: ``16000``)
        n_filter (int, optional): Number of linear filters to apply. (Default: ``128``)
        n_lfcc (int, optional): Number of lfc coefficients to retain. (Default: ``40``)
        f_min (float, optional): Minimum frequency. (Default: ``0.``)
        f_max (float or None, optional): Maximum frequency. (Default: ``None``)
        dct_type (int, optional): type of DCT (discrete cosine transform) to use. (Default: ``2``)
        norm (str, optional): norm to use. (Default: ``"ortho"``)
        log_lf (bool, optional): whether to use log-lf spectrograms instead of db-scaled. (Default: ``False``)
        speckwargs (dict or None, optional): arguments for Spectrogram. (Default: ``None``)

    Example
        >>> waveform, sample_rate = torchaudio.load("test.wav", normalize=True)
        >>> transform = transforms.LFCC(
        >>>     sample_rate=sample_rate,
        >>>     n_lfcc=13,
        >>>     speckwargs={"n_fft": 400, "hop_length": 160, "center": False},
        >>> )
        >>> lfcc = transform(waveform)

    See also:
        :py:func:`torchaudio.functional.linear_fbanks` - The function used to
        generate the filter banks.
    """
    __constants__ = ['sample_rate', 'n_filter', 'n_lfcc', 'dct_type', 'top_db', 'log_lf']

    def __init__(self, sample_rate: int=16000, n_filter: int=128, f_min: float=0.0, f_max: Optional[float]=None, n_lfcc: int=40, dct_type: int=2, norm: str='ortho', log_lf: bool=False, speckwargs: Optional[dict]=None) -> None:
        super(LFCC, self).__init__()
        supported_dct_types = [2]
        if dct_type not in supported_dct_types:
            raise ValueError('DCT type not supported: {}'.format(dct_type))
        self.sample_rate = sample_rate
        self.f_min = f_min
        self.f_max = f_max if f_max is not None else float(sample_rate // 2)
        self.n_filter = n_filter
        self.n_lfcc = n_lfcc
        self.dct_type = dct_type
        self.norm = norm
        self.top_db = 80.0
        self.amplitude_to_DB = AmplitudeToDB('power', self.top_db)
        speckwargs = speckwargs or {}
        self.Spectrogram = Spectrogram(**speckwargs)
        if self.n_lfcc > self.Spectrogram.n_fft:
            raise ValueError('Cannot select more LFCC coefficients than # fft bins')
        filter_mat = F.linear_fbanks(n_freqs=self.Spectrogram.n_fft // 2 + 1, f_min=self.f_min, f_max=self.f_max, n_filter=self.n_filter, sample_rate=self.sample_rate)
        self.register_buffer('filter_mat', filter_mat)
        dct_mat = F.create_dct(self.n_lfcc, self.n_filter, self.norm)
        self.register_buffer('dct_mat', dct_mat)
        self.log_lf = log_lf

    def forward(self, waveform: Tensor) -> Tensor:
        """
        Args:
            waveform (Tensor): Tensor of audio of dimension (..., time).

        Returns:
            Tensor: Linear Frequency Cepstral Coefficients of size (..., ``n_lfcc``, time).
        """
        specgram = self.Spectrogram(waveform)
        specgram = torch.matmul(specgram.transpose(-1, -2), self.filter_mat).transpose(-1, -2)
        if self.log_lf:
            log_offset = 1e-06
            specgram = torch.log(specgram + log_offset)
        else:
            specgram = self.amplitude_to_DB(specgram)
        lfcc = torch.matmul(specgram.transpose(-1, -2), self.dct_mat).transpose(-1, -2)
        return lfcc