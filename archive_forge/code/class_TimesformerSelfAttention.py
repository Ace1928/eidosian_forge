import collections
from typing import Optional, Tuple, Union
import torch
import torch.nn.functional
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_outputs import BaseModelOutput, ImageClassifierOutput
from ...modeling_utils import PreTrainedModel
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from .configuration_timesformer import TimesformerConfig
class TimesformerSelfAttention(nn.Module):

    def __init__(self, config: TimesformerConfig):
        super().__init__()
        num_heads = config.num_attention_heads
        qkv_bias = config.qkv_bias
        attention_dropout_prob = config.attention_probs_dropout_prob
        self.num_heads = num_heads
        head_dim = config.hidden_size // num_heads
        self.scale = head_dim ** (-0.5)
        self.qkv = nn.Linear(config.hidden_size, config.hidden_size * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attention_dropout_prob)

    def forward(self, hidden_states, output_attentions: bool=False):
        batch_size, hidden_size, num_channels = hidden_states.shape
        qkv = self.qkv(hidden_states).reshape(batch_size, hidden_size, 3, self.num_heads, num_channels // self.num_heads).permute(2, 0, 3, 1, 4)
        query, key, value = (qkv[0], qkv[1], qkv[2])
        attention_probs = query @ key.transpose(-2, -1) * self.scale
        attention_probs = attention_probs.softmax(dim=-1)
        attention_probs = self.attn_drop(attention_probs)
        context_layer = (attention_probs @ value).transpose(1, 2).reshape(batch_size, hidden_size, num_channels)
        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        return outputs