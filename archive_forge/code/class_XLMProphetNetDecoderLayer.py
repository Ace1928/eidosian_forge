import copy
import math
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import Tensor, nn
from torch.nn import LayerNorm
from ...activations import ACT2FN
from ...modeling_outputs import BaseModelOutput
from ...modeling_utils import PreTrainedModel
from ...utils import (
from .configuration_xlm_prophetnet import XLMProphetNetConfig
class XLMProphetNetDecoderLayer(nn.Module):
    """
    Decoder block for XLMProphetnet
    """

    def __init__(self, config: XLMProphetNetConfig):
        super().__init__()
        self.self_attn = XLMProphetNetNgramSelfAttention(config)
        self.self_attn_layer_norm = LayerNorm(config.hidden_size)
        if config.add_cross_attention:
            self.cross_attn = XLMProphetNetAttention(config, config.num_decoder_attention_heads)
            self.cross_attn_layer_norm = LayerNorm(config.hidden_size)
        self.feed_forward = XLMProphetNetFeedForward(config, config.decoder_ffn_dim)
        self.feed_forward_layer_norm = LayerNorm(config.hidden_size)

    def forward(self, hidden_states, attention_mask=None, encoder_hidden_states=None, encoder_attn_mask=None, layer_head_mask=None, cross_attn_layer_head_mask=None, extended_predict_attention_mask=None, main_relative_position_buckets=None, predict_relative_position_buckets=None, position_ids=None, past_key_value=None, use_cache: bool=True, output_attentions: bool=False):
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        ngram_attention_output, self_attn_weights, self_attn_weights_ngram, present_key_value = self.self_attn(hidden_states=hidden_states, past_key_value=self_attn_past_key_value, attention_mask=attention_mask, layer_head_mask=layer_head_mask, extended_predict_attention_mask=extended_predict_attention_mask, main_relative_position_buckets=main_relative_position_buckets, predict_relative_position_buckets=predict_relative_position_buckets, position_ids=position_ids)
        hidden_states = self.self_attn_layer_norm(hidden_states + ngram_attention_output)
        cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
        cross_attn_weights = None
        if encoder_hidden_states is not None:
            attention_output, cross_attn_weights, cross_attn_present_key_value = self.cross_attn(hidden_states=hidden_states, key_value_states=encoder_hidden_states, attention_mask=encoder_attn_mask, layer_head_mask=cross_attn_layer_head_mask, past_key_value=cross_attn_past_key_value, output_attentions=output_attentions)
            hidden_states = self.cross_attn_layer_norm(attention_output + hidden_states)
            present_key_value = present_key_value + cross_attn_present_key_value
        feed_forward_output = self.feed_forward(hidden_states)
        hidden_states = self.feed_forward_layer_norm(feed_forward_output + hidden_states)
        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights, self_attn_weights_ngram, cross_attn_weights)
        if use_cache:
            outputs += (present_key_value,)
        return outputs