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
class PerceiverEncoder(nn.Module):
    """The Perceiver Encoder: a scalable, fully attentional encoder."""

    def __init__(self, config, kv_dim=None):
        super().__init__()
        self.config = config
        if config.d_latents % config.num_self_attention_heads != 0:
            raise ValueError(f'num_z_channels ({config.d_latents}) must be divisible by num_self_attend_heads ({config.num_self_attention_heads}).')
        if config.d_latents % config.num_cross_attention_heads != 0:
            raise ValueError(f'num_z_channels ({config.d_latents}) must be divisible by num_cross_attend_heads ({config.num_cross_attention_heads}).')
        self.cross_attention = PerceiverLayer(config, is_cross_attention=True, qk_channels=config.qk_channels, v_channels=config.v_channels, num_heads=config.num_cross_attention_heads, q_dim=config.d_latents, kv_dim=kv_dim, widening_factor=config.cross_attention_widening_factor, use_query_residual=config.use_query_residual)
        self_attention_layers = []
        for _ in range(config.num_self_attends_per_block):
            layer = PerceiverLayer(config, is_cross_attention=False, qk_channels=config.qk_channels, v_channels=config.v_channels, num_heads=config.num_self_attention_heads, q_dim=config.d_latents, kv_dim=config.d_latents, widening_factor=config.self_attention_widening_factor)
            self_attention_layers.append(layer)
        self.self_attends = nn.ModuleList(self_attention_layers)

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.FloatTensor]=None, head_mask: Optional[torch.FloatTensor]=None, inputs: Optional[torch.FloatTensor]=None, inputs_mask: Optional[torch.FloatTensor]=None, output_attentions: Optional[bool]=False, output_hidden_states: Optional[bool]=False, return_dict: Optional[bool]=True) -> Union[Tuple, BaseModelOutputWithCrossAttentions]:
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions else None
        layer_outputs = self.cross_attention(hidden_states, attention_mask=attention_mask, head_mask=None, inputs=inputs, inputs_mask=inputs_mask, output_attentions=output_attentions)
        hidden_states = layer_outputs[0]
        if output_attentions:
            all_cross_attentions = all_cross_attentions + (layer_outputs[1],)
        for _ in range(self.config.num_blocks):
            for i, layer_module in enumerate(self.self_attends):
                if output_hidden_states:
                    all_hidden_states = all_hidden_states + (hidden_states,)
                layer_head_mask = head_mask[i] if head_mask is not None else None
                layer_outputs = layer_module(hidden_states, attention_mask=attention_mask, head_mask=layer_head_mask, output_attentions=output_attentions)
                hidden_states = layer_outputs[0]
                if output_attentions:
                    all_self_attentions = all_self_attentions + (layer_outputs[1],)
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
        if not return_dict:
            return tuple((v for v in [hidden_states, all_hidden_states, all_self_attentions, all_cross_attentions] if v is not None))
        return BaseModelOutputWithCrossAttentions(last_hidden_state=hidden_states, hidden_states=all_hidden_states, attentions=all_self_attentions, cross_attentions=all_cross_attentions)