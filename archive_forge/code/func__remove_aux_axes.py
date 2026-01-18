from typing import List, Optional, Tuple
import torch
from torch import nn, Tensor
from torchaudio._internal import load_state_dict_from_url
from torchaudio.models import wav2vec2_model, Wav2Vec2Model, wavlm_model
def _remove_aux_axes(state_dict, axes):
    for key in ['aux.weight', 'aux.bias']:
        mat = state_dict[key]
        state_dict[key] = torch.stack([mat[i] for i in range(mat.size(0)) if i not in axes])