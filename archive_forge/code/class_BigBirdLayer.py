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
class BigBirdLayer(nn.Module):

    def __init__(self, config, seed=None):
        super().__init__()
        self.config = config
        self.attention_type = config.attention_type
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = BigBirdAttention(config, seed=seed)
        self.is_decoder = config.is_decoder
        self.add_cross_attention = config.add_cross_attention
        if self.add_cross_attention:
            if not self.is_decoder:
                raise TypeError(f'{self} should be used as a decoder model if cross attention is added')
            self.crossattention = BigBirdAttention(config)
        self.intermediate = BigBirdIntermediate(config)
        self.output = BigBirdOutput(config)

    def set_attention_type(self, value: str):
        if value not in ['original_full', 'block_sparse']:
            raise ValueError(f"attention_type can only be set to either 'original_full' or 'block_sparse', but is {value}")
        if value == self.attention_type:
            return
        self.attention_type = value
        self.attention.set_attention_type(value)
        if self.add_cross_attention:
            self.crossattention.set_attention_type(value)

    def forward(self, hidden_states, attention_mask=None, head_mask=None, encoder_hidden_states=None, encoder_attention_mask=None, band_mask=None, from_mask=None, to_mask=None, blocked_encoder_mask=None, past_key_value=None, output_attentions=False):
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        self_attention_outputs = self.attention(hidden_states, attention_mask, head_mask, encoder_hidden_states=encoder_hidden_states, encoder_attention_mask=encoder_attention_mask, past_key_value=self_attn_past_key_value, output_attentions=output_attentions, band_mask=band_mask, from_mask=from_mask, to_mask=to_mask, from_blocked_mask=blocked_encoder_mask, to_blocked_mask=blocked_encoder_mask)
        attention_output = self_attention_outputs[0]
        if self.is_decoder:
            outputs = self_attention_outputs[1:-1]
            present_key_value = self_attention_outputs[-1]
        else:
            outputs = self_attention_outputs[1:]
        cross_attn_present_key_value = None
        if self.is_decoder and encoder_hidden_states is not None:
            if not hasattr(self, 'crossattention'):
                raise ValueError(f'If `encoder_hidden_states` are passed, {self} has to be instantiated with                     cross-attention layers by setting `config.add_cross_attention=True`')
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            cross_attention_outputs = self.crossattention(attention_output, attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask, cross_attn_past_key_value, output_attentions)
            attention_output = cross_attention_outputs[0]
            outputs = outputs + cross_attention_outputs[1:-1]
            cross_attn_present_key_value = cross_attention_outputs[-1]
            present_key_value = present_key_value + cross_attn_present_key_value
        layer_output = apply_chunking_to_forward(self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output)
        outputs = (layer_output,) + outputs
        if self.is_decoder:
            outputs = outputs + (present_key_value,)
        return outputs

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output