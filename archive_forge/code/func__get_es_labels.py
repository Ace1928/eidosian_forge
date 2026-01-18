from typing import List, Optional, Tuple
import torch
from torch import nn, Tensor
from torchaudio._internal import load_state_dict_from_url
from torchaudio.models import wav2vec2_model, Wav2Vec2Model, wavlm_model
def _get_es_labels():
    return ('|', 'e', 'a', 'o', 's', 'n', 'r', 'i', 'l', 'd', 'c', 't', 'u', 'p', 'm', 'b', 'q', 'y', 'g', 'v', 'h', 'ó', 'f', 'í', 'á', 'j', 'z', 'ñ', 'é', 'x', 'ú', 'k', 'w', 'ü')