from typing import Callable, Optional
import torch
from torchaudio.prototype.functional import barkscale_fbanks, chroma_filterbank
from torchaudio.transforms import Spectrogram
class ChromaScale(torch.nn.Module):
    """Converts spectrogram to chromagram.

    .. devices:: CPU CUDA

    .. properties:: Autograd

    Args:
        sample_rate (int): Sample rate of audio signal.
        n_freqs (int): Number of frequency bins in STFT. See ``n_fft`` in :class:`Spectrogram`.
        n_chroma (int, optional): Number of chroma. (Default: ``12``)
        tuning (float, optional): Tuning deviation from A440 in fractions of a chroma bin. (Default: 0.0)
        ctroct (float, optional): Center of Gaussian dominance window to weight filters by, in octaves. (Default: 5.0)
        octwidth (float or None, optional): Width of Gaussian dominance window to weight filters by, in octaves.
            If ``None``, then disable weighting altogether. (Default: 2.0)
        norm (int, optional): order of norm to normalize filter bank by. (Default: 2)
        base_c (bool, optional): If True, then start filter bank at C. Otherwise, start at A. (Default: True)

    Example
        >>> waveform, sample_rate = torchaudio.load("test.wav", normalize=True)
        >>> spectrogram_transform = transforms.Spectrogram(n_fft=1024)
        >>> spectrogram = spectrogram_transform(waveform)
        >>> chroma_transform = transforms.ChromaScale(sample_rate=sample_rate, n_freqs=1024 // 2 + 1)
        >>> chroma_spectrogram = chroma_transform(spectrogram)

    See also:
        :py:func:`torchaudio.prototype.functional.chroma_filterbank` — function used to
        generate the filter bank.
    """

    def __init__(self, sample_rate: int, n_freqs: int, *, n_chroma: int=12, tuning: float=0.0, ctroct: float=5.0, octwidth: Optional[float]=2.0, norm: int=2, base_c: bool=True):
        super().__init__()
        fb = chroma_filterbank(sample_rate, n_freqs, n_chroma, tuning=tuning, ctroct=ctroct, octwidth=octwidth, norm=norm, base_c=base_c)
        self.register_buffer('fb', fb)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            specgram (torch.Tensor): Spectrogram of dimension (..., ``n_freqs``, time).

        Returns:
            torch.Tensor: Chroma spectrogram of size (..., ``n_chroma``, time).
        """
        return torch.matmul(x.transpose(-1, -2), self.fb).transpose(-1, -2)