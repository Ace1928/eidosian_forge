import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
from torch import Tensor
from torchaudio._internal import load_state_dict_from_url
from torchaudio.functional import mu_law_decoding
from torchaudio.models import Tacotron2, WaveRNN
from torchaudio.transforms import GriffinLim, InverseMelScale
from . import utils
from .interface import Tacotron2TTSBundle
class _GriffinLimVocoder(torch.nn.Module, Tacotron2TTSBundle.Vocoder):

    def __init__(self):
        super().__init__()
        self._sample_rate = 22050
        self._inv_mel = InverseMelScale(n_stft=1024 // 2 + 1, n_mels=80, sample_rate=self.sample_rate, f_min=0.0, f_max=8000.0, mel_scale='slaney', norm='slaney')
        self._griffin_lim = GriffinLim(n_fft=1024, power=1, hop_length=256, win_length=1024)

    @property
    def sample_rate(self):
        return self._sample_rate

    def forward(self, mel_spec, lengths=None):
        mel_spec = torch.exp(mel_spec)
        mel_spec = mel_spec.clone().detach().requires_grad_(True)
        spec = self._inv_mel(mel_spec)
        spec = spec.detach().requires_grad_(False)
        waveforms = self._griffin_lim(spec)
        return (waveforms, lengths)