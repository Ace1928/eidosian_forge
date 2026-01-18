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
class NllbMoeDecoderLayer(nn.Module):

    def __init__(self, config: NllbMoeConfig, is_sparse: bool=False):
        super().__init__()
        self.embed_dim = config.d_model
        self.is_sparse = is_sparse
        self.self_attn = NllbMoeAttention(embed_dim=self.embed_dim, num_heads=config.decoder_attention_heads, dropout=config.attention_dropout, is_decoder=True)
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.attn_dropout = nn.Dropout(config.dropout)
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.cross_attention = NllbMoeAttention(self.embed_dim, config.decoder_attention_heads, config.attention_dropout, is_decoder=True)
        self.cross_attention_layer_norm = nn.LayerNorm(self.embed_dim)
        if not self.is_sparse:
            self.ffn = NllbMoeDenseActDense(config, ffn_dim=config.decoder_ffn_dim)
        else:
            self.ffn = NllbMoeSparseMLP(config, ffn_dim=config.decoder_ffn_dim)
        self.ff_layer_norm = nn.LayerNorm(config.d_model)
        self.ff_dropout = nn.Dropout(config.activation_dropout)

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor]=None, encoder_hidden_states: Optional[torch.Tensor]=None, encoder_attention_mask: Optional[torch.Tensor]=None, layer_head_mask: Optional[torch.Tensor]=None, cross_attn_layer_head_mask: Optional[torch.Tensor]=None, past_key_value: Optional[Tuple[torch.Tensor]]=None, output_attentions: Optional[bool]=False, output_router_logits: Optional[bool]=False, use_cache: Optional[bool]=True) -> torch.Tensor:
        """
        Args:
            hidden_states (`torch.FloatTensor`):
                input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`):
                attention mask of size `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very
                large negative values.
            encoder_hidden_states (`torch.FloatTensor`):
                cross attention input to the layer of shape `(batch, seq_len, embed_dim)`
            encoder_attention_mask (`torch.FloatTensor`):
                encoder attention mask of size `(batch, 1, tgt_len, src_len)` where padding elements are indicated by
                very large negative values.
            layer_head_mask (`torch.FloatTensor`):
                mask for attention heads in a given layer of size `(encoder_attention_heads,)`.
            cross_attn_layer_head_mask (`torch.FloatTensor`):
                mask for cross-attention heads in a given layer of size `(decoder_attention_heads,)`.
            past_key_value (`Tuple(torch.FloatTensor)`):
                cached past key and value projection states
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        hidden_states, self_attn_weights, present_key_value = self.self_attn(hidden_states=hidden_states, past_key_value=self_attn_past_key_value, attention_mask=attention_mask, layer_head_mask=layer_head_mask, output_attentions=output_attentions)
        hidden_states = self.attn_dropout(hidden_states)
        hidden_states = residual + hidden_states
        cross_attn_present_key_value = None
        cross_attn_weights = None
        if encoder_hidden_states is not None:
            residual = hidden_states
            hidden_states = self.cross_attention_layer_norm(hidden_states)
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            hidden_states, cross_attn_weights, cross_attn_present_key_value = self.cross_attention(hidden_states=hidden_states, encoder_hidden_states=encoder_hidden_states, past_key_value=cross_attn_past_key_value, attention_mask=encoder_attention_mask, layer_head_mask=cross_attn_layer_head_mask, output_attentions=output_attentions)
            hidden_states = self.attn_dropout(hidden_states)
            hidden_states = residual + hidden_states
            present_key_value += cross_attn_present_key_value
        residual = hidden_states
        hidden_states = self.ff_layer_norm(hidden_states)
        if self.is_sparse:
            hidden_states, router_states = self.ffn(hidden_states, attention_mask)
        else:
            hidden_states, router_states = (self.ffn(hidden_states), None)
        hidden_states = self.ff_dropout(hidden_states)
        hidden_states = residual + hidden_states
        if hidden_states.dtype == torch.float16 and torch.isinf(hidden_states).any():
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        outputs = (hidden_states, present_key_value)
        if output_attentions:
            outputs += (self_attn_weights, cross_attn_weights)
        if output_router_logits:
            outputs += (router_states,)
        return outputs