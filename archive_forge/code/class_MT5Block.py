import copy
import math
import os
import warnings
from typing import List, Optional, Tuple, Union
import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import (
from ...utils.model_parallel_utils import assert_device_map, get_device_map
from .configuration_mt5 import MT5Config
class MT5Block(nn.Module):

    def __init__(self, config, has_relative_attention_bias=False):
        super().__init__()
        self.is_decoder = config.is_decoder
        self.layer = nn.ModuleList()
        self.layer.append(MT5LayerSelfAttention(config, has_relative_attention_bias=has_relative_attention_bias))
        if self.is_decoder:
            self.layer.append(MT5LayerCrossAttention(config))
        self.layer.append(MT5LayerFF(config))

    def forward(self, hidden_states, attention_mask=None, position_bias=None, encoder_hidden_states=None, encoder_attention_mask=None, encoder_decoder_position_bias=None, layer_head_mask=None, cross_attn_layer_head_mask=None, past_key_value=None, use_cache=False, output_attentions=False, return_dict=True):
        if past_key_value is not None:
            if not self.is_decoder:
                logger.warning('`past_key_values` is passed to the encoder. Please make sure this is intended.')
            expected_num_past_key_values = 2 if encoder_hidden_states is None else 4
            if len(past_key_value) != expected_num_past_key_values:
                raise ValueError(f'There should be {expected_num_past_key_values} past states. {('2 (past / key) for cross attention. ' if expected_num_past_key_values == 4 else '')}Got {len(past_key_value)} past key / value states')
            self_attn_past_key_value = past_key_value[:2]
            cross_attn_past_key_value = past_key_value[2:]
        else:
            self_attn_past_key_value, cross_attn_past_key_value = (None, None)
        self_attention_outputs = self.layer[0](hidden_states, attention_mask=attention_mask, position_bias=position_bias, layer_head_mask=layer_head_mask, past_key_value=self_attn_past_key_value, use_cache=use_cache, output_attentions=output_attentions)
        hidden_states, present_key_value_state = self_attention_outputs[:2]
        attention_outputs = self_attention_outputs[2:]
        if hidden_states.dtype == torch.float16:
            clamp_value = torch.where(torch.isinf(hidden_states).any(), torch.finfo(hidden_states.dtype).max - 1000, torch.finfo(hidden_states.dtype).max)
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        do_cross_attention = self.is_decoder and encoder_hidden_states is not None
        if do_cross_attention:
            if present_key_value_state is not None:
                query_length = present_key_value_state[0].shape[2]
            else:
                query_length = None
            cross_attention_outputs = self.layer[1](hidden_states, key_value_states=encoder_hidden_states, attention_mask=encoder_attention_mask, position_bias=encoder_decoder_position_bias, layer_head_mask=cross_attn_layer_head_mask, past_key_value=cross_attn_past_key_value, query_length=query_length, use_cache=use_cache, output_attentions=output_attentions)
            hidden_states = cross_attention_outputs[0]
            if hidden_states.dtype == torch.float16:
                clamp_value = torch.where(torch.isinf(hidden_states).any(), torch.finfo(hidden_states.dtype).max - 1000, torch.finfo(hidden_states.dtype).max)
                hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
            if present_key_value_state is not None:
                present_key_value_state = present_key_value_state + cross_attention_outputs[1]
            attention_outputs = attention_outputs + cross_attention_outputs[2:]
        hidden_states = self.layer[-1](hidden_states)
        if hidden_states.dtype == torch.float16:
            clamp_value = torch.where(torch.isinf(hidden_states).any(), torch.finfo(hidden_states.dtype).max - 1000, torch.finfo(hidden_states.dtype).max)
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        outputs = (hidden_states,)
        if use_cache:
            outputs = outputs + (present_key_value_state,) + attention_outputs
        else:
            outputs = outputs + attention_outputs
        return outputs