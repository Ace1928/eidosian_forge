import math
import warnings
from typing import Optional, Tuple, Union
import numpy as np
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss
from ...activations import ACT2FN
from ...integrations.deepspeed import is_deepspeed_zero3_enabled
from ...modeling_attn_mask_utils import _prepare_4d_attention_mask
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...utils import (
from .configuration_wav2vec2_bert import Wav2Vec2BertConfig
class Wav2Vec2BertEncoderLayer(nn.Module):
    """Conformer block based on https://arxiv.org/abs/2005.08100."""

    def __init__(self, config):
        super().__init__()
        embed_dim = config.hidden_size
        dropout = config.attention_dropout
        self.ffn1_layer_norm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)
        self.ffn1 = Wav2Vec2BertFeedForward(config)
        self.self_attn_layer_norm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)
        self.self_attn_dropout = nn.Dropout(dropout)
        self.self_attn = Wav2Vec2BertSelfAttention(config)
        self.conv_module = Wav2Vec2BertConvolutionModule(config)
        self.ffn2_layer_norm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)
        self.ffn2 = Wav2Vec2BertFeedForward(config)
        self.final_layer_norm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)

    def forward(self, hidden_states, attention_mask: Optional[torch.Tensor]=None, relative_position_embeddings: Optional[torch.Tensor]=None, output_attentions: bool=False, conv_attention_mask: Optional[torch.Tensor]=None):
        hidden_states = hidden_states
        residual = hidden_states
        hidden_states = self.ffn1_layer_norm(hidden_states)
        hidden_states = self.ffn1(hidden_states)
        hidden_states = hidden_states * 0.5 + residual
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states, attn_weigts = self.self_attn(hidden_states=hidden_states, attention_mask=attention_mask, relative_position_embeddings=relative_position_embeddings, output_attentions=output_attentions)
        hidden_states = self.self_attn_dropout(hidden_states)
        hidden_states = hidden_states + residual
        residual = hidden_states
        hidden_states = self.conv_module(hidden_states, attention_mask=conv_attention_mask)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.ffn2_layer_norm(hidden_states)
        hidden_states = self.ffn2(hidden_states)
        hidden_states = hidden_states * 0.5 + residual
        hidden_states = self.final_layer_norm(hidden_states)
        return (hidden_states, attn_weigts)