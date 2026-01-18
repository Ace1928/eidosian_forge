import math
from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from ...activations import ACT2FN
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import ALL_LAYERNORM_LAYERS
from ...utils import (
from .configuration_pix2struct import Pix2StructConfig, Pix2StructTextConfig, Pix2StructVisionConfig
class Pix2StructVisionAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.key_value_proj_dim = config.d_kv
        self.n_heads = config.num_attention_heads
        self.dropout = config.attention_dropout
        self.inner_dim = self.n_heads * self.key_value_proj_dim
        self.query = nn.Linear(self.hidden_size, self.inner_dim, bias=False)
        self.key = nn.Linear(self.hidden_size, self.inner_dim, bias=False)
        self.value = nn.Linear(self.hidden_size, self.inner_dim, bias=False)
        self.output = nn.Linear(self.inner_dim, self.hidden_size, bias=False)
        self.gradient_checkpointing = False

    def forward(self, hidden_states, attention_mask=None, position_bias=None, layer_head_mask=None, output_attentions=False):
        """
        Self-attention block
        """
        batch_size, seq_length = hidden_states.shape[:2]

        def to_projection_shape(states):
            """projection"""
            return states.contiguous().view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
        query_states = to_projection_shape(self.query(hidden_states))
        key_states = to_projection_shape(self.key(hidden_states))
        value_states = to_projection_shape(self.value(hidden_states))
        scores = torch.matmul(query_states, key_states.transpose(3, 2))
        if position_bias is None:
            position_bias = torch.zeros((1, self.n_heads, seq_length, seq_length), device=scores.device, dtype=scores.dtype)
            if self.gradient_checkpointing and self.training:
                position_bias.requires_grad = True
            if attention_mask is None:
                attention_mask = torch.ones((batch_size, seq_length), device=scores.device, dtype=scores.dtype)
            if attention_mask.dim() == 2:
                position_bias = position_bias + attention_mask[:, None, None, :].to(position_bias.device)
            else:
                position_bias = position_bias + attention_mask.to(position_bias.device)
            position_bias = 1 - position_bias
        position_bias_masked = position_bias.masked_fill(position_bias == 1, torch.finfo(scores.dtype).min)
        scores += position_bias_masked
        scores = torch.max(scores, torch.tensor(torch.finfo(scores.dtype).min))
        attn_weights = nn.functional.softmax(scores, dim=-1, dtype=torch.float32).type_as(scores)
        attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        if layer_head_mask is not None:
            attn_weights = attn_weights * layer_head_mask
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)
        attn_output = self.output(attn_output)
        outputs = (attn_output,) + (position_bias,)
        if output_attentions:
            outputs = outputs + (attn_weights,)
        return outputs