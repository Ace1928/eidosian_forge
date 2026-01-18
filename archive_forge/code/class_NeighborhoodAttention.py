import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_outputs import BackboneOutput
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import (
from ...utils.backbone_utils import BackboneMixin
from .configuration_dinat import DinatConfig
class NeighborhoodAttention(nn.Module):

    def __init__(self, config, dim, num_heads, kernel_size, dilation):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f'The hidden size ({dim}) is not a multiple of the number of attention heads ({num_heads})')
        self.num_attention_heads = num_heads
        self.attention_head_size = int(dim / num_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.rpb = nn.Parameter(torch.zeros(num_heads, 2 * self.kernel_size - 1, 2 * self.kernel_size - 1))
        self.query = nn.Linear(self.all_head_size, self.all_head_size, bias=config.qkv_bias)
        self.key = nn.Linear(self.all_head_size, self.all_head_size, bias=config.qkv_bias)
        self.value = nn.Linear(self.all_head_size, self.all_head_size, bias=config.qkv_bias)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 3, 1, 2, 4)

    def forward(self, hidden_states: torch.Tensor, output_attentions: Optional[bool]=False) -> Tuple[torch.Tensor]:
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = query_layer / math.sqrt(self.attention_head_size)
        attention_scores = natten2dqkrpb(query_layer, key_layer, self.rpb, self.kernel_size, self.dilation)
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        context_layer = natten2dav(attention_probs, value_layer, self.kernel_size, self.dilation)
        context_layer = context_layer.permute(0, 2, 3, 1, 4).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)
        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        return outputs