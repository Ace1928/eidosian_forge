import math
from typing import List, Optional, Tuple, Union
import numpy as np
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, L1Loss
from ...activations import ACT2FN
from ...integrations.deepspeed import is_deepspeed_zero3_enabled
from ...modeling_attn_mask_utils import _prepare_4d_attention_mask, _prepare_4d_causal_attention_mask
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from .configuration_speecht5 import SpeechT5Config, SpeechT5HifiGanConfig
class SpeechT5ScaledPositionalEncoding(nn.Module):
    """
    Scaled positional encoding, see §3.2 in https://arxiv.org/abs/1809.08895
    """

    def __init__(self, dropout, dim, max_len=5000):
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2, dtype=torch.int64).float() * -(math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        pe = pe.unsqueeze(0)
        super().__init__()
        self.register_buffer('pe', pe, persistent=False)
        self.dropout = nn.Dropout(p=dropout)
        self.dim = dim
        self.alpha = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, emb):
        emb = emb + self.alpha * self.pe[:, :emb.size(1)]
        emb = self.dropout(emb)
        return emb