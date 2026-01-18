from typing import Callable, Optional
import torch
from torchaudio.prototype.functional import barkscale_fbanks, chroma_filterbank
from torchaudio.transforms import Spectrogram
class BarkScale(torch.nn.Module):
    """Turn a normal STFT into a bark frequency STFT with triangular filter banks.

    .. devices:: CPU CUDA

    .. properties:: Autograd TorchScript

    Args:
        n_barks (int, optional): Number of bark filterbanks. (Default: ``128``)
        sample_rate (int, optional): Sample rate of audio signal. (Default: ``16000``)
        f_min (float, optional): Minimum frequency. (Default: ``0.``)
        f_max (float or None, optional): Maximum frequency. (Default: ``sample_rate // 2``)
        n_stft (int, optional): Number of bins in STFT. See ``n_fft`` in :class:`Spectrogram`. (Default: ``201``)
        norm (str or None, optional): If ``"slaney"``, divide the triangular bark weights by the width of the bark band
            (area normalization). (Default: ``None``)
        bark_scale (str, optional): Scale to use: ``traunmuller``, ``schroeder`` or ``wang``. (Default: ``traunmuller``)

    Example
        >>> waveform, sample_rate = torchaudio.load("test.wav", normalize=True)
        >>> spectrogram_transform = transforms.Spectrogram(n_fft=1024)
        >>> spectrogram = spectrogram_transform(waveform)
        >>> barkscale_transform = transforms.BarkScale(sample_rate=sample_rate, n_stft=1024 // 2 + 1)
        >>> barkscale_spectrogram = barkscale_transform(spectrogram)

    See also:
        :py:func:`torchaudio.prototype.functional.barkscale_fbanks` - The function used to
        generate the filter banks.
    """
    __constants__ = ['n_barks', 'sample_rate', 'f_min', 'f_max']

    def __init__(self, n_barks: int=128, sample_rate: int=16000, f_min: float=0.0, f_max: Optional[float]=None, n_stft: int=201, bark_scale: str='traunmuller') -> None:
        super(BarkScale, self).__init__()
        self.n_barks = n_barks
        self.sample_rate = sample_rate
        self.f_max = f_max if f_max is not None else float(sample_rate // 2)
        self.f_min = f_min
        self.bark_scale = bark_scale
        if f_min > self.f_max:
            raise ValueError('Require f_min: {} <= f_max: {}'.format(f_min, self.f_max))
        fb = barkscale_fbanks(n_stft, self.f_min, self.f_max, self.n_barks, self.sample_rate, self.bark_scale)
        self.register_buffer('fb', fb)

    def forward(self, specgram: torch.Tensor) -> torch.Tensor:
        """
        Args:
            specgram (torch.Tensor): A spectrogram STFT of dimension (..., freq, time).

        Returns:
            torch.Tensor: Bark frequency spectrogram of size (..., ``n_barks``, time).
        """
        bark_specgram = torch.matmul(specgram.transpose(-1, -2), self.fb).transpose(-1, -2)
        return bark_specgram