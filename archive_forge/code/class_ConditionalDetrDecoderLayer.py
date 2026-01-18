import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import torch
from torch import Tensor, nn
from ...activations import ACT2FN
from ...modeling_attn_mask_utils import _prepare_4d_attention_mask
from ...modeling_outputs import BaseModelOutput, BaseModelOutputWithCrossAttentions, Seq2SeqModelOutput
from ...modeling_utils import PreTrainedModel
from ...utils import (
from ...utils.backbone_utils import load_backbone
from .configuration_conditional_detr import ConditionalDetrConfig
class ConditionalDetrDecoderLayer(nn.Module):

    def __init__(self, config: ConditionalDetrConfig):
        super().__init__()
        self.embed_dim = config.d_model
        d_model = config.d_model
        self.sa_qcontent_proj = nn.Linear(d_model, d_model)
        self.sa_qpos_proj = nn.Linear(d_model, d_model)
        self.sa_kcontent_proj = nn.Linear(d_model, d_model)
        self.sa_kpos_proj = nn.Linear(d_model, d_model)
        self.sa_v_proj = nn.Linear(d_model, d_model)
        self.self_attn = ConditionalDetrAttention(embed_dim=self.embed_dim, out_dim=self.embed_dim, num_heads=config.decoder_attention_heads, dropout=config.attention_dropout)
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.ca_qcontent_proj = nn.Linear(d_model, d_model)
        self.ca_qpos_proj = nn.Linear(d_model, d_model)
        self.ca_kcontent_proj = nn.Linear(d_model, d_model)
        self.ca_kpos_proj = nn.Linear(d_model, d_model)
        self.ca_v_proj = nn.Linear(d_model, d_model)
        self.ca_qpos_sine_proj = nn.Linear(d_model, d_model)
        self.encoder_attn = ConditionalDetrAttention(self.embed_dim * 2, self.embed_dim, config.decoder_attention_heads, dropout=config.attention_dropout)
        self.encoder_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.fc1 = nn.Linear(self.embed_dim, config.decoder_ffn_dim)
        self.fc2 = nn.Linear(config.decoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)
        self.nhead = config.decoder_attention_heads

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor]=None, object_queries: Optional[torch.Tensor]=None, query_position_embeddings: Optional[torch.Tensor]=None, query_sine_embed: Optional[torch.Tensor]=None, encoder_hidden_states: Optional[torch.Tensor]=None, encoder_attention_mask: Optional[torch.Tensor]=None, output_attentions: Optional[bool]=False, is_first: Optional[bool]=False, **kwargs):
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(seq_len, batch, embed_dim)`
            attention_mask (`torch.FloatTensor`): attention mask of size
                `(batch, 1, target_len, source_len)` where padding elements are indicated by very large negative
                values.
            object_queries (`torch.FloatTensor`, *optional*):
                object_queries that are added to the queries and keys
            in the cross-attention layer.
            query_position_embeddings (`torch.FloatTensor`, *optional*):
                object_queries that are added to the queries and keys
            in the self-attention layer.
            encoder_hidden_states (`torch.FloatTensor`):
                cross attention input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_attention_mask (`torch.FloatTensor`): encoder attention mask of size
                `(batch, 1, target_len, source_len)` where padding elements are indicated by very large negative
                values.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        position_embeddings = kwargs.pop('position_embeddings', None)
        if kwargs:
            raise ValueError(f'Unexpected arguments {kwargs.keys()}')
        if position_embeddings is not None and object_queries is not None:
            raise ValueError('Cannot specify both position_embeddings and object_queries. Please use just object_queries')
        if position_embeddings is not None:
            logger.warning_once('position_embeddings has been deprecated and will be removed in v4.34. Please use object_queries instead')
            object_queries = position_embeddings
        residual = hidden_states
        q_content = self.sa_qcontent_proj(hidden_states)
        q_pos = self.sa_qpos_proj(query_position_embeddings)
        k_content = self.sa_kcontent_proj(hidden_states)
        k_pos = self.sa_kpos_proj(query_position_embeddings)
        v = self.sa_v_proj(hidden_states)
        _, num_queries, n_model = q_content.shape
        q = q_content + q_pos
        k = k_content + k_pos
        hidden_states, self_attn_weights = self.self_attn(hidden_states=q, attention_mask=attention_mask, key_states=k, value_states=v, output_attentions=output_attentions)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        q_content = self.ca_qcontent_proj(hidden_states)
        k_content = self.ca_kcontent_proj(encoder_hidden_states)
        v = self.ca_v_proj(encoder_hidden_states)
        batch_size, num_queries, n_model = q_content.shape
        _, source_len, _ = k_content.shape
        k_pos = self.ca_kpos_proj(object_queries)
        if is_first:
            q_pos = self.ca_qpos_proj(query_position_embeddings)
            q = q_content + q_pos
            k = k_content + k_pos
        else:
            q = q_content
            k = k_content
        q = q.view(batch_size, num_queries, self.nhead, n_model // self.nhead)
        query_sine_embed = self.ca_qpos_sine_proj(query_sine_embed)
        query_sine_embed = query_sine_embed.view(batch_size, num_queries, self.nhead, n_model // self.nhead)
        q = torch.cat([q, query_sine_embed], dim=3).view(batch_size, num_queries, n_model * 2)
        k = k.view(batch_size, source_len, self.nhead, n_model // self.nhead)
        k_pos = k_pos.view(batch_size, source_len, self.nhead, n_model // self.nhead)
        k = torch.cat([k, k_pos], dim=3).view(batch_size, source_len, n_model * 2)
        cross_attn_weights = None
        if encoder_hidden_states is not None:
            residual = hidden_states
            hidden_states, cross_attn_weights = self.encoder_attn(hidden_states=q, attention_mask=encoder_attention_mask, key_states=k, value_states=v, output_attentions=output_attentions)
            hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
            hidden_states = residual + hidden_states
            hidden_states = self.encoder_attn_layer_norm(hidden_states)
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