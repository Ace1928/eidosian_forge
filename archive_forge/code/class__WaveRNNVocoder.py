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
class _WaveRNNVocoder(torch.nn.Module, Tacotron2TTSBundle.Vocoder):

    def __init__(self, model: WaveRNN, min_level_db: Optional[float]=-100):
        super().__init__()
        self._sample_rate = 22050
        self._model = model
        self._min_level_db = min_level_db

    @property
    def sample_rate(self):
        return self._sample_rate

    def forward(self, mel_spec, lengths=None):
        mel_spec = torch.exp(mel_spec)
        mel_spec = 20 * torch.log10(torch.clamp(mel_spec, min=1e-05))
        if self._min_level_db is not None:
            mel_spec = (self._min_level_db - mel_spec) / self._min_level_db
            mel_spec = torch.clamp(mel_spec, min=0, max=1)
        waveform, lengths = self._model.infer(mel_spec, lengths)
        waveform = utils._unnormalize_waveform(waveform, self._model.n_bits)
        waveform = mu_law_decoding(waveform, self._model.n_classes)
        waveform = waveform.squeeze(1)
        return (waveform, lengths)