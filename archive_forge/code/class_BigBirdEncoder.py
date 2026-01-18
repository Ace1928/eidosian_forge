import math
import os
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import numpy as np
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import apply_chunking_to_forward
from ...utils import (
from .configuration_big_bird import BigBirdConfig
class BigBirdEncoder(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.attention_type = config.attention_type
        self.layer = nn.ModuleList([BigBirdLayer(config, seed=layer_idx) for layer_idx in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    def set_attention_type(self, value: str):
        if value not in ['original_full', 'block_sparse']:
            raise ValueError(f"attention_type can only be set to either 'original_full' or 'block_sparse', but is {value}")
        if value == self.attention_type:
            return
        self.attention_type = value
        for layer in self.layer:
            layer.set_attention_type(value)

    def forward(self, hidden_states, attention_mask=None, head_mask=None, encoder_hidden_states=None, encoder_attention_mask=None, past_key_values=None, use_cache=None, output_attentions=False, output_hidden_states=False, band_mask=None, from_mask=None, to_mask=None, blocked_encoder_mask=None, return_dict=True) -> Union[BaseModelOutputWithPastAndCrossAttentions, Tuple]:
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None
        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once('`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`...')
                use_cache = False
        next_decoder_cache = () if use_cache else None
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None
            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(layer_module.__call__, hidden_states, attention_mask, layer_head_mask, encoder_hidden_states, encoder_attention_mask, band_mask, from_mask, to_mask, blocked_encoder_mask, past_key_value, output_attentions)
            else:
                layer_outputs = layer_module(hidden_states, attention_mask, layer_head_mask, encoder_hidden_states, encoder_attention_mask, band_mask, from_mask, to_mask, blocked_encoder_mask, past_key_value, output_attentions)
            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        if not return_dict:
            return tuple((v for v in [hidden_states, next_decoder_cache, all_hidden_states, all_self_attentions, all_cross_attentions] if v is not None))
        return BaseModelOutputWithPastAndCrossAttentions(last_hidden_state=hidden_states, past_key_values=next_decoder_cache, hidden_states=all_hidden_states, attentions=all_self_attentions, cross_attentions=all_cross_attentions)