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
class CvtAttention(nn.Module):

    def __init__(self, num_heads, embed_dim, kernel_size, padding_q, padding_kv, stride_q, stride_kv, qkv_projection_method, qkv_bias, attention_drop_rate, drop_rate, with_cls_token=True):
        super().__init__()
        self.attention = CvtSelfAttention(num_heads, embed_dim, kernel_size, padding_q, padding_kv, stride_q, stride_kv, qkv_projection_method, qkv_bias, attention_drop_rate, with_cls_token)
        self.output = CvtSelfOutput(embed_dim, drop_rate)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(heads, self.attention.num_attention_heads, self.attention.attention_head_size, self.pruned_heads)
        self.attention.query = prune_linear_layer(self.attention.query, index)
        self.attention.key = prune_linear_layer(self.attention.key, index)
        self.attention.value = prune_linear_layer(self.attention.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)
        self.attention.num_attention_heads = self.attention.num_attention_heads - len(heads)
        self.attention.all_head_size = self.attention.attention_head_size * self.attention.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(self, hidden_state, height, width):
        self_output = self.attention(hidden_state, height, width)
        attention_output = self.output(self_output, hidden_state)
        return attention_output