import math
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
from torch import Tensor, nn
from torch.nn import CrossEntropyLoss, LayerNorm
from ...activations import ACT2FN
from ...integrations.deepspeed import is_deepspeed_zero3_enabled
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...utils import (
from .configuration_fsmt import FSMTConfig
class DecoderLayer(nn.Module):

    def __init__(self, config: FSMTConfig):
        super().__init__()
        self.embed_dim = config.d_model
        self.self_attn = Attention(embed_dim=self.embed_dim, num_heads=config.decoder_attention_heads, dropout=config.attention_dropout)
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout
        self.self_attn_layer_norm = LayerNorm(self.embed_dim)
        self.encoder_attn = Attention(self.embed_dim, config.decoder_attention_heads, dropout=config.attention_dropout, encoder_decoder_attention=True)
        self.encoder_attn_layer_norm = LayerNorm(self.embed_dim)
        self.fc1 = nn.Linear(self.embed_dim, config.decoder_ffn_dim)
        self.fc2 = nn.Linear(config.decoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = LayerNorm(self.embed_dim)

    def forward(self, x, encoder_hidden_states, encoder_attn_mask=None, layer_state=None, causal_mask=None, layer_head_mask=None, cross_attn_layer_head_mask=None, decoder_padding_mask=None, output_attentions=False):
        residual = x
        if layer_state is None:
            layer_state = {}
        x, self_attn_weights = self.self_attn(query=x, key=x, layer_state=layer_state, key_padding_mask=decoder_padding_mask, attn_mask=causal_mask, layer_head_mask=layer_head_mask, output_attentions=output_attentions)
        x = nn.functional.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.self_attn_layer_norm(x)
        residual = x
        assert self.encoder_attn.cache_key != self.self_attn.cache_key
        x, cross_attn_weights = self.encoder_attn(query=x, key=encoder_hidden_states, key_padding_mask=encoder_attn_mask, layer_state=layer_state, layer_head_mask=cross_attn_layer_head_mask, output_attentions=output_attentions)
        x = nn.functional.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.encoder_attn_layer_norm(x)
        residual = x
        x = self.activation_fn(self.fc1(x))
        x = nn.functional.dropout(x, p=self.activation_dropout, training=self.training)
        x = self.fc2(x)
        x = nn.functional.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.final_layer_norm(x)
        return (x, self_attn_weights, layer_state, cross_attn_weights)