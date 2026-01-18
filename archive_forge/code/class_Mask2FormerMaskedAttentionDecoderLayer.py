import math
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
from torch import Tensor, nn
from ...activations import ACT2FN
from ...file_utils import (
from ...modeling_outputs import BaseModelOutput, BaseModelOutputWithCrossAttentions
from ...modeling_utils import PreTrainedModel
from ...utils import is_accelerate_available, logging
from ...utils.backbone_utils import load_backbone
from .configuration_mask2former import Mask2FormerConfig
class Mask2FormerMaskedAttentionDecoderLayer(nn.Module):
    """
    The Mask2FormerMaskedAttentionDecoderLayer is made up of self-attention, cross (masked) attention as well as FFN
    blocks. The cross attention block used as part of `Mask2FormerMaskedAttentionDecoderLayer` is actually a `masked
    attention` block that restricts the attention to localized features centered around predicted segments which leads
    to faster convergence and improved performance. The order of self and cross (i.e. masked) attention blocks have
    also been swapped in Mask2FormerMaskedAttentionDecoder compared to a standard DetrDecoder as an optimization
    improvement.

    Args:
        config (`Mask2FormerConfig`):
            The configuration used to initialize the Mask2FormerMaskedAttentionDecoder.
    """

    def __init__(self, config: Mask2FormerConfig):
        super().__init__()
        self.config = config
        self.embed_dim = self.config.hidden_dim
        self.pre_norm = self.config.pre_norm
        self.self_attn = Mask2FormerAttention(embed_dim=self.embed_dim, num_heads=config.num_attention_heads, dropout=config.dropout, is_decoder=True)
        self.dropout = self.config.dropout
        self.activation_fn = ACT2FN[self.config.activation_function]
        self.activation_dropout = self.config.dropout
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.cross_attn = nn.MultiheadAttention(self.embed_dim, self.config.num_attention_heads, self.config.dropout)
        self.cross_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.fc1 = nn.Linear(self.embed_dim, self.config.dim_feedforward)
        self.fc2 = nn.Linear(self.config.dim_feedforward, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, hidden_states: torch.Tensor, level_index: int=None, attention_mask: Optional[torch.Tensor]=None, position_embeddings: Optional[torch.Tensor]=None, query_position_embeddings: Optional[torch.Tensor]=None, encoder_hidden_states: Optional[torch.Tensor]=None, encoder_attention_mask: Optional[torch.Tensor]=None, output_attentions: Optional[bool]=False):
        cross_attn_weights = None
        self_attn_weights = None
        residual = hidden_states
        hidden_states, cross_attn_weights = self.cross_attn(query=self.with_pos_embed(hidden_states, query_position_embeddings), key=self.with_pos_embed(encoder_hidden_states[level_index], position_embeddings[level_index]), value=encoder_hidden_states[level_index], attn_mask=encoder_attention_mask, key_padding_mask=None)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.cross_attn_layer_norm(hidden_states)
        residual = hidden_states
        hidden_states, self_attn_weights = self.self_attn(hidden_states=hidden_states, position_embeddings=query_position_embeddings, attention_mask=None, output_attentions=True)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights, cross_attn_weights)
        return outputs

    def forward_pre(self, hidden_states: torch.Tensor, level_index: int=None, attention_mask: Optional[torch.Tensor]=None, position_embeddings: Optional[torch.Tensor]=None, query_position_embeddings: Optional[torch.Tensor]=None, encoder_hidden_states: Optional[torch.Tensor]=None, encoder_attention_mask: Optional[torch.Tensor]=None, output_attentions: Optional[bool]=False):
        cross_attn_weights = None
        self_attn_weights = None
        residual = hidden_states
        hidden_states = self.cross_attn_layer_norm(hidden_states)
        hidden_states, cross_attn_weights = self.cross_attn(query=self.with_pos_embed(hidden_states, query_position_embeddings), key=self.with_pos_embed(encoder_hidden_states[level_index], position_embeddings[level_index]), value=encoder_hidden_states[level_index], attn_mask=encoder_attention_mask, key_padding_mask=None)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states, self_attn_weights = self.self_attn(hidden_states=hidden_states, position_embeddings=query_position_embeddings, attention_mask=None, output_attentions=True)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights, cross_attn_weights)
        return outputs

    def forward(self, hidden_states: torch.Tensor, level_index: int=None, attention_mask: Optional[torch.Tensor]=None, position_embeddings: Optional[torch.Tensor]=None, query_position_embeddings: Optional[torch.Tensor]=None, encoder_hidden_states: Optional[torch.Tensor]=None, encoder_attention_mask: Optional[torch.Tensor]=None, output_attentions: Optional[bool]=False):
        """
        Args:
            hidden_states (`torch.FloatTensor`):
                Input to the layer of shape `(seq_len, batch, embed_dim)`.
            attention_mask (`torch.FloatTensor`):
                Attention mask of shape `(1, seq_len, tgt_len, src_len)`.
            position_embeddings (`torch.FloatTensor`, *optional*):
                Position embeddings that are added to the keys in the masked-attention layer.
            query_position_embeddings (`torch.FloatTensor`, *optional*):
                Position embeddings that are added to the queries and keys in the self-attention layer.
            encoder_hidden_states (`torch.FloatTensor`):
                Cross attention input to the layer of shape `(seq_len, batch, embed_dim)`.
            encoder_attention_mask (`torch.FloatTensor`):
                Encoder attention mask of size`(1, seq_len, tgt_len, src_len)`.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        if self.pre_norm:
            outputs = self.forward_pre(hidden_states=hidden_states, level_index=level_index, position_embeddings=position_embeddings, query_position_embeddings=query_position_embeddings, encoder_hidden_states=encoder_hidden_states, encoder_attention_mask=encoder_attention_mask, output_attentions=output_attentions)
        else:
            outputs = self.forward_post(hidden_states=hidden_states, level_index=level_index, position_embeddings=position_embeddings, query_position_embeddings=query_position_embeddings, encoder_hidden_states=encoder_hidden_states, encoder_attention_mask=encoder_attention_mask, output_attentions=output_attentions)
        return outputs