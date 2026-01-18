from typing import List, Optional, Tuple
import torch
from torch import nn, Tensor
from torchaudio._internal import load_state_dict_from_url
from torchaudio.models import wav2vec2_model, Wav2Vec2Model, wavlm_model
def _get_de_labels():
    return ('|', 'e', 'n', 'i', 'r', 's', 't', 'a', 'd', 'h', 'u', 'l', 'g', 'c', 'm', 'o', 'b', 'w', 'f', 'k', 'z', 'p', 'v', 'ü', 'ä', 'ö', 'j', 'ß', 'y', 'x', 'q')