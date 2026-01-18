import abc
import math
from dataclasses import dataclass
from functools import reduce
from operator import __add__
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple, Union
import numpy as np
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_outputs import BaseModelOutputWithCrossAttentions
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import apply_chunking_to_forward, find_pruneable_heads_and_indices, meshgrid, prune_linear_layer
from ...utils import (
from .configuration_perceiver import PerceiverConfig
class PerceiverLayer(nn.Module):

    def __init__(self, config, is_cross_attention=False, qk_channels=None, v_channels=None, num_heads=1, q_dim=None, kv_dim=None, widening_factor=4, use_query_residual=True):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = PerceiverAttention(config, is_cross_attention=is_cross_attention, qk_channels=qk_channels, v_channels=v_channels, num_heads=num_heads, q_dim=q_dim, kv_dim=kv_dim, use_query_residual=use_query_residual)
        self.layernorm = nn.LayerNorm(q_dim)
        self.mlp = PerceiverMLP(config, input_size=q_dim, widening_factor=widening_factor)

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.FloatTensor]=None, head_mask: Optional[torch.FloatTensor]=None, inputs: Optional[torch.FloatTensor]=None, inputs_mask: Optional[torch.FloatTensor]=None, output_attentions: Optional[bool]=False) -> Tuple[torch.Tensor]:
        attention_outputs = self.attention(hidden_states, attention_mask, head_mask, inputs, inputs_mask, output_attentions)
        attention_output = attention_outputs[0]
        outputs = attention_outputs[1:]
        layer_output = apply_chunking_to_forward(self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output)
        layer_output = layer_output + attention_output
        outputs = (layer_output,) + outputs
        return outputs

    def feed_forward_chunk(self, attention_output):
        layer_output = self.layernorm(attention_output)
        layer_output = self.mlp(layer_output)
        return layer_output