import copy
import math
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
from torch import Tensor, nn
from torch.cuda.amp import autocast
from ...activations import ACT2FN
from ...modeling_outputs import BaseModelOutput
from ...modeling_utils import PreTrainedModel
from ...utils import (
from ...utils.backbone_utils import load_backbone
from .configuration_oneformer import OneFormerConfig
class OneFormerTextEncoder(nn.Module):

    def __init__(self, context_length: int, width: int, layers: int, vocab_size, use_checkpoint=False, layer_norm_eps=1e-05):
        super().__init__()
        heads = width // 64
        self.context_length = context_length
        self.width = width
        self.transformer = OneFormerTextTransformer(width=width, layers=layers, heads=heads, attn_mask=self.build_attention_mask(), use_checkpoint=use_checkpoint, layer_norm_eps=layer_norm_eps)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, width))
        self.ln_final = nn.LayerNorm(width, eps=layer_norm_eps)
        self.token_embedding = nn.Embedding(vocab_size, width)

    def build_attention_mask(self):
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float('-inf'))
        mask.triu_(1)
        return mask

    def forward(self, text):
        hidden_state = self.token_embedding(text)
        hidden_state = hidden_state + self.positional_embedding
        hidden_state = hidden_state.permute(1, 0, 2)
        hidden_state = self.transformer(hidden_state)
        hidden_state = hidden_state.permute(1, 0, 2)
        hidden_state = self.ln_final(hidden_state)
        hidden_state = hidden_state[torch.arange(hidden_state.shape[0]), text.argmax(dim=-1)]
        return hidden_state