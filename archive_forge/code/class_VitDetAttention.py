import collections.abc
import math
from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from ...activations import ACT2FN
from ...modeling_outputs import BackboneOutput, BaseModelOutput
from ...modeling_utils import PreTrainedModel
from ...utils import (
from ...utils.backbone_utils import BackboneMixin
from .configuration_vitdet import VitDetConfig
class VitDetAttention(nn.Module):
    """Multi-head Attention block with relative position embeddings."""

    def __init__(self, config, input_size=None):
        """
        Args:
            config (`VitDetConfig`):
                Model configuration.
            input_size (`Tuple[int]`, *optional*):
                Input resolution, only required in case relative position embeddings are added.
        """
        super().__init__()
        dim = config.hidden_size
        num_heads = config.num_attention_heads
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** (-0.5)
        self.qkv = nn.Linear(dim, dim * 3, bias=config.qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.use_relative_position_embeddings = config.use_relative_position_embeddings
        if self.use_relative_position_embeddings:
            self.rel_pos_h = nn.Parameter(torch.zeros(2 * input_size[0] - 1, head_dim))
            self.rel_pos_w = nn.Parameter(torch.zeros(2 * input_size[1] - 1, head_dim))

    def forward(self, hidden_state, output_attentions=False):
        batch_size, height, width, _ = hidden_state.shape
        qkv = self.qkv(hidden_state).reshape(batch_size, height * width, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        queries, keys, values = qkv.reshape(3, batch_size * self.num_heads, height * width, -1).unbind(0)
        attention_scores = queries * self.scale @ keys.transpose(-2, -1)
        if self.use_relative_position_embeddings:
            attention_scores = add_decomposed_relative_positions(attention_scores, queries, self.rel_pos_h, self.rel_pos_w, (height, width), (height, width))
        attention_probs = attention_scores.softmax(dim=-1)
        hidden_state = attention_probs @ values
        hidden_state = hidden_state.view(batch_size, self.num_heads, height, width, -1)
        hidden_state = hidden_state.permute(0, 2, 3, 1, 4)
        hidden_state = hidden_state.reshape(batch_size, height, width, -1)
        hidden_state = self.proj(hidden_state)
        if output_attentions:
            attention_probs = attention_probs.reshape(batch_size, self.num_heads, attention_probs.shape[-2], attention_probs.shape[-1])
            outputs = (hidden_state, attention_probs)
        else:
            outputs = (hidden_state,)
        return outputs