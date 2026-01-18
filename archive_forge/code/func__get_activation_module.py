import math
from typing import List, Optional, Tuple
import torch
from torchaudio.models.emformer import _EmformerAttention, _EmformerImpl, _get_weight_init_gains
def _get_activation_module(activation: str) -> torch.nn.Module:
    if activation == 'relu':
        return torch.nn.ReLU()
    elif activation == 'gelu':
        return torch.nn.GELU()
    elif activation == 'silu':
        return torch.nn.SiLU()
    else:
        raise ValueError(f'Unsupported activation {activation}')