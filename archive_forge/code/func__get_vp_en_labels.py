from typing import List, Optional, Tuple
import torch
from torch import nn, Tensor
from torchaudio._internal import load_state_dict_from_url
from torchaudio.models import wav2vec2_model, Wav2Vec2Model, wavlm_model
def _get_vp_en_labels():
    return ('|', 'e', 't', 'o', 'i', 'a', 'n', 's', 'r', 'h', 'l', 'd', 'c', 'u', 'm', 'p', 'f', 'g', 'w', 'y', 'b', 'v', 'k', 'x', 'j', 'q', 'z')