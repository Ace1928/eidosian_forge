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
class CvtLayer(nn.Module):
    """
    CvtLayer composed by attention layers, normalization and multi-layer perceptrons (mlps).
    """

    def __init__(self, num_heads, embed_dim, kernel_size, padding_q, padding_kv, stride_q, stride_kv, qkv_projection_method, qkv_bias, attention_drop_rate, drop_rate, mlp_ratio, drop_path_rate, with_cls_token=True):
        super().__init__()
        self.attention = CvtAttention(num_heads, embed_dim, kernel_size, padding_q, padding_kv, stride_q, stride_kv, qkv_projection_method, qkv_bias, attention_drop_rate, drop_rate, with_cls_token)
        self.intermediate = CvtIntermediate(embed_dim, mlp_ratio)
        self.output = CvtOutput(embed_dim, mlp_ratio, drop_rate)
        self.drop_path = CvtDropPath(drop_prob=drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()
        self.layernorm_before = nn.LayerNorm(embed_dim)
        self.layernorm_after = nn.LayerNorm(embed_dim)

    def forward(self, hidden_state, height, width):
        self_attention_output = self.attention(self.layernorm_before(hidden_state), height, width)
        attention_output = self_attention_output
        attention_output = self.drop_path(attention_output)
        hidden_state = attention_output + hidden_state
        layer_output = self.layernorm_after(hidden_state)
        layer_output = self.intermediate(layer_output)
        layer_output = self.output(layer_output, hidden_state)
        layer_output = self.drop_path(layer_output)
        return layer_output