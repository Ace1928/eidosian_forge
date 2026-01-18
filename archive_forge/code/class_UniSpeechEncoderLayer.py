import math
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import numpy as np
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss
from ...activations import ACT2FN
from ...integrations.deepspeed import is_deepspeed_zero3_enabled
from ...modeling_outputs import BaseModelOutput, CausalLMOutput, SequenceClassifierOutput, Wav2Vec2BaseModelOutput
from ...modeling_utils import PreTrainedModel
from ...utils import (
from .configuration_unispeech import UniSpeechConfig
class UniSpeechEncoderLayer(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.attention = UniSpeechAttention(embed_dim=config.hidden_size, num_heads=config.num_attention_heads, dropout=config.attention_dropout, is_decoder=False)
        self.dropout = nn.Dropout(config.hidden_dropout)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.feed_forward = UniSpeechFeedForward(config)
        self.final_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states, attention_mask=None, output_attentions=False):
        attn_residual = hidden_states
        hidden_states, attn_weights, _ = self.attention(hidden_states, attention_mask=attention_mask, output_attentions=output_attentions)
        hidden_states = self.dropout(hidden_states)
        hidden_states = attn_residual + hidden_states
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = hidden_states + self.feed_forward(hidden_states)
        hidden_states = self.final_layer_norm(hidden_states)
        outputs = (hidden_states,)
        if output_attentions:
            outputs += (attn_weights,)
        return outputs