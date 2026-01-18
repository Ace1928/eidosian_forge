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
class PerceiverAttention(nn.Module):
    """Attention module, including a dense block."""

    def __init__(self, config, is_cross_attention=False, qk_channels=None, v_channels=None, num_heads=1, q_dim=None, kv_dim=None, use_query_residual=True):
        super().__init__()
        if is_cross_attention and qk_channels is None:
            if config.cross_attention_shape_for_attention == 'q':
                qk_channels = q_dim
            elif config.cross_attention_shape_for_attention == 'kv':
                qk_channels = kv_dim
            else:
                raise ValueError(f'Unknown value {config.cross_attention_shape_for_attention} for cross_attention_shape_for_attention.')
        else:
            if qk_channels is None:
                qk_channels = q_dim
            if v_channels is None:
                v_channels = qk_channels
        self.self = PerceiverSelfAttention(config, is_cross_attention=is_cross_attention, qk_channels=qk_channels, v_channels=v_channels, num_heads=num_heads, q_dim=q_dim, kv_dim=kv_dim)
        output_channels = None
        if is_cross_attention:
            output_channels = q_dim
        elif output_channels is None:
            output_channels = v_channels
        self.output = PerceiverSelfOutput(config, input_channels=self.self.v_channels, output_channels=output_channels)
        self.use_query_residual = use_query_residual
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads)
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.FloatTensor]=None, head_mask: Optional[torch.FloatTensor]=None, inputs: Optional[torch.FloatTensor]=None, inputs_mask: Optional[torch.FloatTensor]=None, output_attentions: Optional[bool]=False) -> Tuple[torch.Tensor]:
        self_outputs = self.self(hidden_states, attention_mask, head_mask, inputs, inputs_mask, output_attentions)
        attention_output = self.output(self_outputs[0])
        if self.use_query_residual:
            attention_output = attention_output + hidden_states
        outputs = (attention_output,) + self_outputs[1:]
        return outputs