import collections.abc
import math
from copy import deepcopy
from dataclasses import dataclass
from typing import Optional, Set, Tuple, Union
import numpy as np
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_outputs import BaseModelOutput, ImageClassifierOutput
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import (
from ...utils.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from .configuration_videomae import VideoMAEConfig
class VideoMAESelfAttention(nn.Module):

    def __init__(self, config: VideoMAEConfig) -> None:
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and (not hasattr(config, 'embedding_size')):
            raise ValueError(f'The hidden size {(config.hidden_size,)} is not a multiple of the number of attention heads {config.num_attention_heads}.')
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.query = nn.Linear(config.hidden_size, self.all_head_size, bias=False)
        self.key = nn.Linear(config.hidden_size, self.all_head_size, bias=False)
        self.value = nn.Linear(config.hidden_size, self.all_head_size, bias=False)
        if config.qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(self.all_head_size))
            self.v_bias = nn.Parameter(torch.zeros(self.all_head_size))
        else:
            self.q_bias = None
            self.v_bias = None
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, head_mask: Optional[torch.Tensor]=None, output_attentions: bool=False) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        k_bias = torch.zeros_like(self.v_bias, requires_grad=False) if self.q_bias is not None else None
        keys = nn.functional.linear(input=hidden_states, weight=self.key.weight, bias=k_bias)
        values = nn.functional.linear(input=hidden_states, weight=self.value.weight, bias=self.v_bias)
        queries = nn.functional.linear(input=hidden_states, weight=self.query.weight, bias=self.q_bias)
        key_layer = self.transpose_for_scores(keys)
        value_layer = self.transpose_for_scores(values)
        query_layer = self.transpose_for_scores(queries)
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        if head_mask is not None:
            attention_probs = attention_probs * head_mask
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)
        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        return outputs