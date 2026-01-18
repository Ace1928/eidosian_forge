import math
from typing import Optional, Tuple, Union
import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN, gelu
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward, logging
from .configuration_mpnet import MPNetConfig
class MPNetAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.attn = MPNetSelfAttention(config)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(heads, self.attn.num_attention_heads, self.attn.attention_head_size, self.pruned_heads)
        self.attn.q = prune_linear_layer(self.attn.q, index)
        self.attn.k = prune_linear_layer(self.attn.k, index)
        self.attn.v = prune_linear_layer(self.attn.v, index)
        self.attn.o = prune_linear_layer(self.attn.o, index, dim=1)
        self.attn.num_attention_heads = self.attn.num_attention_heads - len(heads)
        self.attn.all_head_size = self.attn.attention_head_size * self.attn.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(self, hidden_states, attention_mask=None, head_mask=None, position_bias=None, output_attentions=False, **kwargs):
        self_outputs = self.attn(hidden_states, attention_mask, head_mask, position_bias, output_attentions=output_attentions)
        attention_output = self.LayerNorm(self.dropout(self_outputs[0]) + hidden_states)
        outputs = (attention_output,) + self_outputs[1:]
        return outputs