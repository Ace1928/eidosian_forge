from dataclasses import dataclass
from typing import Any, Dict, Optional
import torch
import torch.nn.functional as F
from torch.nn import Module
from torchaudio._internal import load_state_dict_from_url
from torchaudio.prototype.models.hifi_gan import hifigan_vocoder, HiFiGANVocoder
from torchaudio.transforms import MelSpectrogram
class _HiFiGANMelSpectrogram(torch.nn.Module):
    """
    Generate mel spectrogram in a way equivalent to the original HiFiGAN implementation:
    https://github.com/jik876/hifi-gan/blob/4769534d45265d52a904b850da5a622601885777/meldataset.py#L49-L72

    This class wraps around :py:class:`torchaudio.transforms.MelSpectrogram`, but performs extra steps to achive
    equivalence with the HiFiGAN implementation.

    Args:
        hop_size (int): Length of hop between STFT windows.
        n_fft (int): Size of FFT, creates ``n_fft // 2 + 1`` bins.
        win_length (int): Window size.
        f_min (float or None):  Minimum frequency.
        f_max (float or None): Maximum frequency.
        sample_rate (int):  Sample rate of audio signal.
        n_mels (int):  Number of mel filterbanks.
    """

    def __init__(self, hop_size: int, n_fft: int, win_length: int, f_min: Optional[float], f_max: Optional[float], sample_rate: float, n_mels: int):
        super(_HiFiGANMelSpectrogram, self).__init__()
        self.mel_transform = MelSpectrogram(sample_rate=sample_rate, n_fft=n_fft, win_length=win_length, hop_length=hop_size, f_min=f_min, f_max=f_max, n_mels=n_mels, normalized=False, pad=0, mel_scale='slaney', norm='slaney', center=False)
        self.sample_rate = sample_rate
        self.hop_size = hop_size
        self.n_fft = n_fft
        self.win_length = win_length
        self.f_min = f_min
        self.f_max = f_max
        self.n_mels = n_mels
        self.pad_size = int((n_fft - hop_size) / 2)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """Generate mel spectrogram from a waveform. Should have same sample rate as ``self.sample_rate``.

        Args:
            waveform (Tensor): waveform of shape ``(batch_size, time_length)``.
        Returns:
            Tensor of shape ``(batch_size, n_mel, time_length)``
        """
        ref_waveform = F.pad(waveform.unsqueeze(1), (self.pad_size, self.pad_size), mode='reflect')
        ref_waveform = ref_waveform.squeeze(1)
        spectr = (self.mel_transform.spectrogram(ref_waveform) + 1e-09) ** 0.5
        mel_spectrogram = self.mel_transform.mel_scale(spectr)
        mel_spectrogram = torch.log(torch.clamp(mel_spectrogram, min=1e-05))
        return mel_spectrogram