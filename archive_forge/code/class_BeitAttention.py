import collections.abc
import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import Tensor, nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import find_pruneable_heads_and_indices, meshgrid, prune_linear_layer
from ...utils import (
from ...utils.backbone_utils import BackboneMixin
from .configuration_beit import BeitConfig
class BeitAttention(nn.Module):

    def __init__(self, config: BeitConfig, window_size: Optional[tuple]=None) -> None:
        super().__init__()
        self.attention = BeitSelfAttention(config, window_size=window_size)
        self.output = BeitSelfOutput(config)
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

    def forward(self, hidden_states: torch.Tensor, head_mask: Optional[torch.Tensor]=None, output_attentions: bool=False, relative_position_bias: Optional['BeitRelativePositionBias']=None) -> Union[Tuple[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        self_outputs = self.attention(hidden_states, head_mask, output_attentions, relative_position_bias)
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]
        return outputs