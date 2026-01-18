import math
from typing import List, Optional, Tuple, Union
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from ...activations import ACT2FN
from ...integrations.deepspeed import is_deepspeed_zero3_enabled
from ...modeling_attn_mask_utils import _prepare_4d_attention_mask, _prepare_4d_causal_attention_mask
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...utils import (
from .configuration_nllb_moe import NllbMoeConfig
class NllbMoeEncoderLayer(nn.Module):

    def __init__(self, config: NllbMoeConfig, is_sparse: bool=False):
        super().__init__()
        self.embed_dim = config.d_model
        self.is_sparse = is_sparse
        self.self_attn = NllbMoeAttention(embed_dim=self.embed_dim, num_heads=config.encoder_attention_heads, dropout=config.attention_dropout)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        if not self.is_sparse:
            self.ffn = NllbMoeDenseActDense(config, ffn_dim=config.encoder_ffn_dim)
        else:
            self.ffn = NllbMoeSparseMLP(config, ffn_dim=config.encoder_ffn_dim)
        self.ff_layer_norm = nn.LayerNorm(config.d_model)
        self.ff_dropout = nn.Dropout(config.activation_dropout)

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor, layer_head_mask: torch.Tensor, output_attentions: bool=False, output_router_logits: bool=False) -> torch.Tensor:
        """
        Args:
            hidden_states (`torch.FloatTensor`):
                input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`):
                attention mask of size `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very
                large negative values.
            layer_head_mask (`torch.FloatTensor`): mask for attention heads in a given layer of size
                `(encoder_attention_heads,)`.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states, attn_weights, _ = self.self_attn(hidden_states=hidden_states, attention_mask=attention_mask, layer_head_mask=layer_head_mask, output_attentions=output_attentions)
        hidden_states = self.attn_dropout(hidden_states)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.ff_layer_norm(hidden_states)
        if self.is_sparse:
            hidden_states, router_states = self.ffn(hidden_states, attention_mask)
        else:
            hidden_states, router_states = (self.ffn(hidden_states), None)
        hidden_states = self.ff_dropout(hidden_states)
        hidden_states = residual + hidden_states
        if hidden_states.dtype == torch.float16 and (torch.isinf(hidden_states).any() or torch.isnan(hidden_states).any()):
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        outputs = (hidden_states,)
        if output_attentions:
            outputs += (attn_weights,)
        if output_router_logits:
            outputs += (router_states,)
        return outputs