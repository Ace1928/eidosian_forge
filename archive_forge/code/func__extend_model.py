from typing import List, Optional, Tuple
import torch
from torch import nn, Tensor
from torchaudio._internal import load_state_dict_from_url
from torchaudio.models import wav2vec2_model, Wav2Vec2Model, wavlm_model
def _extend_model(module, normalize_waveform, apply_log_softmax=False, append_star=False):
    """Add extra transformations to the model"""
    return _Wav2Vec2Model(module, normalize_waveform, apply_log_softmax, append_star)