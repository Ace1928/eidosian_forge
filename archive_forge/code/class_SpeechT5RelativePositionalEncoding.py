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
class SpeechT5RelativePositionalEncoding(torch.nn.Module):

    def __init__(self, dim, max_length=1000):
        super().__init__()
        self.dim = dim
        self.max_length = max_length
        self.pe_k = torch.nn.Embedding(2 * max_length, dim)

    def forward(self, hidden_states):
        seq_len = hidden_states.shape[1]
        pos_seq = torch.arange(0, seq_len).long().to(hidden_states.device)
        pos_seq = pos_seq[:, None] - pos_seq[None, :]
        pos_seq[pos_seq < -self.max_length] = -self.max_length
        pos_seq[pos_seq >= self.max_length] = self.max_length - 1
        pos_seq = pos_seq + self.max_length
        return self.pe_k(pos_seq)