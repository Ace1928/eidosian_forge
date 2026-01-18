import math
from dataclasses import dataclass
from typing import Optional, Tuple
import torch
import torch.utils.checkpoint
from torch import nn
from ...activations import ACT2FN
from ...file_utils import add_start_docstrings, add_start_docstrings_to_model_forward, replace_return_docstrings
from ...modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling, ModelOutput
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import prune_linear_layer
from ...utils import logging
from ...utils.backbone_utils import load_backbone
from .configuration_tvp import TvpConfig
class TvpAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and (not hasattr(config, 'embedding_size')):
            raise ValueError(f'The hidden size {config.hidden_size} is not a multiple of the number of attention heads {config.num_attention_heads}')
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        self.attn_dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        mask = torch.ones(self.num_attention_heads, self.attention_head_size)
        heads = set(heads) - self.pruned_heads
        for head in heads:
            head = head - sum((1 if h < head else 0 for h in self.pruned_heads))
            mask[head] = 0
        mask = mask.view(-1).contiguous().eq(1)
        index = torch.arange(len(mask))[mask].long()
        self.query = prune_linear_layer(self.query, index)
        self.key = prune_linear_layer(self.key, index)
        self.value = prune_linear_layer(self.value, index)
        self.dense = prune_linear_layer(self.dense, index, dim=1)
        self.num_attention_heads = self.num_attention_heads - len(heads)
        self.all_head_size = self.attention_head_size * self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def _reshape(self, tensor: torch.Tensor, sequence_length: int, batch_size: int):
        return tensor.view(batch_size, sequence_length, self.num_attention_heads, self.attention_head_size).transpose(1, 2).contiguous()

    def forward(self, hidden_states, attention_mask=None, head_mask=None, output_attentions: Optional[bool]=None):
        batch_size, sequence_length = hidden_states.shape[:2]
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)
        query_layer = self._reshape(mixed_query_layer, sequence_length, batch_size)
        key_layer = self._reshape(mixed_key_layer, sequence_length, batch_size)
        value_layer = self._reshape(mixed_value_layer, sequence_length, batch_size)
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = self.attn_dropout(attention_probs)
        if head_mask is not None:
            attention_probs = attention_probs * head_mask
        attn_output = torch.matmul(attention_probs, value_layer)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, sequence_length, self.all_head_size)
        attn_output = self.dense(attn_output)
        attn_output = self.dropout(attn_output)
        attn_output = self.layer_norm(attn_output + hidden_states)
        outputs = (attn_output, attention_probs) if output_attentions else (attn_output,)
        return outputs