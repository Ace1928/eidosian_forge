import copy
import math
from typing import List, Optional, Tuple, Union
import numpy as np
import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_attn_mask_utils import _prepare_4d_attention_mask, _prepare_4d_causal_attention_mask
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...utils import (
from .configuration_bigbird_pegasus import BigBirdPegasusConfig
class BigBirdPegasusEncoderLayer(nn.Module):

    def __init__(self, config: BigBirdPegasusConfig, seed=None):
        super().__init__()
        self.attention_type = config.attention_type
        self.embed_dim = config.d_model
        self.self_attn = BigBirdPegasusEncoderAttention(config, seed=seed)
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout
        self.fc1 = nn.Linear(self.embed_dim, config.encoder_ffn_dim)
        self.fc2 = nn.Linear(config.encoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor, layer_head_mask: torch.Tensor, band_mask=None, from_mask=None, to_mask=None, from_blocked_mask=None, to_blocked_mask=None, output_attentions: bool=False):
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        self_attention_outputs = self.self_attn(hidden_states=hidden_states, attention_mask=attention_mask, head_mask=layer_head_mask, output_attentions=output_attentions, band_mask=band_mask, from_mask=from_mask, to_mask=to_mask, from_blocked_mask=from_blocked_mask, to_blocked_mask=to_blocked_mask)
        hidden_states = self_attention_outputs[0]
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        if hidden_states.dtype == torch.float16 and (torch.isinf(hidden_states).any() or torch.isnan(hidden_states).any()):
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attention_outputs[1],)
        return outputs

    def set_attention_type(self, value: str):
        if value not in ['original_full', 'block_sparse']:
            raise ValueError(f"attention_type can only be set to either 'original_full' or 'block_sparse', but is {value}")
        if value == self.attention_type:
            return
        self.attention_type = value
        self.self_attn.set_attention_type(value)