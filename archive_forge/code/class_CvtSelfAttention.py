import collections.abc
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...file_utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward
from ...modeling_outputs import ImageClassifierOutputWithNoAttention, ModelOutput
from ...modeling_utils import PreTrainedModel, find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import logging
from .configuration_cvt import CvtConfig
class CvtSelfAttention(nn.Module):

    def __init__(self, num_heads, embed_dim, kernel_size, padding_q, padding_kv, stride_q, stride_kv, qkv_projection_method, qkv_bias, attention_drop_rate, with_cls_token=True, **kwargs):
        super().__init__()
        self.scale = embed_dim ** (-0.5)
        self.with_cls_token = with_cls_token
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.convolution_projection_query = CvtSelfAttentionProjection(embed_dim, kernel_size, padding_q, stride_q, projection_method='linear' if qkv_projection_method == 'avg' else qkv_projection_method)
        self.convolution_projection_key = CvtSelfAttentionProjection(embed_dim, kernel_size, padding_kv, stride_kv, projection_method=qkv_projection_method)
        self.convolution_projection_value = CvtSelfAttentionProjection(embed_dim, kernel_size, padding_kv, stride_kv, projection_method=qkv_projection_method)
        self.projection_query = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)
        self.projection_key = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)
        self.projection_value = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)
        self.dropout = nn.Dropout(attention_drop_rate)

    def rearrange_for_multi_head_attention(self, hidden_state):
        batch_size, hidden_size, _ = hidden_state.shape
        head_dim = self.embed_dim // self.num_heads
        return hidden_state.view(batch_size, hidden_size, self.num_heads, head_dim).permute(0, 2, 1, 3)

    def forward(self, hidden_state, height, width):
        if self.with_cls_token:
            cls_token, hidden_state = torch.split(hidden_state, [1, height * width], 1)
        batch_size, hidden_size, num_channels = hidden_state.shape
        hidden_state = hidden_state.permute(0, 2, 1).view(batch_size, num_channels, height, width)
        key = self.convolution_projection_key(hidden_state)
        query = self.convolution_projection_query(hidden_state)
        value = self.convolution_projection_value(hidden_state)
        if self.with_cls_token:
            query = torch.cat((cls_token, query), dim=1)
            key = torch.cat((cls_token, key), dim=1)
            value = torch.cat((cls_token, value), dim=1)
        head_dim = self.embed_dim // self.num_heads
        query = self.rearrange_for_multi_head_attention(self.projection_query(query))
        key = self.rearrange_for_multi_head_attention(self.projection_key(key))
        value = self.rearrange_for_multi_head_attention(self.projection_value(value))
        attention_score = torch.einsum('bhlk,bhtk->bhlt', [query, key]) * self.scale
        attention_probs = torch.nn.functional.softmax(attention_score, dim=-1)
        attention_probs = self.dropout(attention_probs)
        context = torch.einsum('bhlt,bhtv->bhlv', [attention_probs, value])
        _, _, hidden_size, _ = context.shape
        context = context.permute(0, 2, 1, 3).contiguous().view(batch_size, hidden_size, self.num_heads * head_dim)
        return context