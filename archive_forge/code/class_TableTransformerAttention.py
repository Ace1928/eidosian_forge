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
from .configuration_table_transformer import TableTransformerConfig
class TableTransformerAttention(nn.Module):
    """
    Multi-headed attention from 'Attention Is All You Need' paper.

    Here, we add position embeddings to the queries and keys (as explained in the TABLE_TRANSFORMER paper).
    """

    def __init__(self, embed_dim: int, num_heads: int, dropout: float=0.0, bias: bool=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        if self.head_dim * num_heads != self.embed_dim:
            raise ValueError(f'embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`: {num_heads}).')
        self.scaling = self.head_dim ** (-0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def _shape(self, tensor: torch.Tensor, seq_len: int, batch_size: int):
        return tensor.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def with_pos_embed(self, tensor: torch.Tensor, object_queries: Optional[Tensor], **kwargs):
        position_embeddings = kwargs.pop('position_embeddings', None)
        if kwargs:
            raise ValueError(f'Unexpected arguments {kwargs.keys()}')
        if position_embeddings is not None and object_queries is not None:
            raise ValueError('Cannot specify both position_embeddings and object_queries. Please use just object_queries')
        if position_embeddings is not None:
            logger.warning_once('position_embeddings has been deprecated and will be removed in v4.34. Please use object_queries instead')
            object_queries = position_embeddings
        return tensor if object_queries is None else tensor + object_queries

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor]=None, object_queries: Optional[torch.Tensor]=None, key_value_states: Optional[torch.Tensor]=None, spatial_position_embeddings: Optional[torch.Tensor]=None, output_attentions: bool=False, **kwargs) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""
        position_embeddings = kwargs.pop('position_ebmeddings', None)
        key_value_position_embeddings = kwargs.pop('key_value_position_embeddings', None)
        if kwargs:
            raise ValueError(f'Unexpected arguments {kwargs.keys()}')
        if position_embeddings is not None and object_queries is not None:
            raise ValueError('Cannot specify both position_embeddings and object_queries. Please use just object_queries')
        if key_value_position_embeddings is not None and spatial_position_embeddings is not None:
            raise ValueError('Cannot specify both key_value_position_embeddings and spatial_position_embeddings. Please use just spatial_position_embeddings')
        if position_embeddings is not None:
            logger.warning_once('position_embeddings has been deprecated and will be removed in v4.34. Please use object_queries instead')
            object_queries = position_embeddings
        if key_value_position_embeddings is not None:
            logger.warning_once('key_value_position_embeddings has been deprecated and will be removed in v4.34. Please use spatial_position_embeddings instead')
            spatial_position_embeddings = key_value_position_embeddings
        is_cross_attention = key_value_states is not None
        batch_size, target_len, embed_dim = hidden_states.size()
        if object_queries is not None:
            hidden_states_original = hidden_states
            hidden_states = self.with_pos_embed(hidden_states, object_queries)
        if spatial_position_embeddings is not None:
            key_value_states_original = key_value_states
            key_value_states = self.with_pos_embed(key_value_states, spatial_position_embeddings)
        query_states = self.q_proj(hidden_states) * self.scaling
        if is_cross_attention:
            key_states = self._shape(self.k_proj(key_value_states), -1, batch_size)
            value_states = self._shape(self.v_proj(key_value_states_original), -1, batch_size)
        else:
            key_states = self._shape(self.k_proj(hidden_states), -1, batch_size)
            value_states = self._shape(self.v_proj(hidden_states_original), -1, batch_size)
        proj_shape = (batch_size * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, target_len, batch_size).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)
        source_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
        if attn_weights.size() != (batch_size * self.num_heads, target_len, source_len):
            raise ValueError(f'Attention weights should be of size {(batch_size * self.num_heads, target_len, source_len)}, but is {attn_weights.size()}')
        if attention_mask is not None:
            if attention_mask.size() != (batch_size, 1, target_len, source_len):
                raise ValueError(f'Attention mask should be of size {(batch_size, 1, target_len, source_len)}, but is {attention_mask.size()}')
            attn_weights = attn_weights.view(batch_size, self.num_heads, target_len, source_len) + attention_mask
            attn_weights = attn_weights.view(batch_size * self.num_heads, target_len, source_len)
        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        if output_attentions:
            attn_weights_reshaped = attn_weights.view(batch_size, self.num_heads, target_len, source_len)
            attn_weights = attn_weights_reshaped.view(batch_size * self.num_heads, target_len, source_len)
        else:
            attn_weights_reshaped = None
        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        attn_output = torch.bmm(attn_probs, value_states)
        if attn_output.size() != (batch_size * self.num_heads, target_len, self.head_dim):
            raise ValueError(f'`attn_output` should be of size {(batch_size, self.num_heads, target_len, self.head_dim)}, but is {attn_output.size()}')
        attn_output = attn_output.view(batch_size, self.num_heads, target_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(batch_size, target_len, embed_dim)
        attn_output = self.out_proj(attn_output)
        return (attn_output, attn_weights_reshaped)