import math
from typing import Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import apply_chunking_to_forward, find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from .configuration_lilt import LiltConfig
class LiltAttention(nn.Module):

    def __init__(self, config, position_embedding_type=None):
        super().__init__()
        self.self = LiltSelfAttention(config, position_embedding_type=position_embedding_type)
        self.output = LiltSelfOutput(config)
        self.pruned_heads = set()
        ori_hidden_size = config.hidden_size
        config.hidden_size = config.hidden_size // config.channel_shrink_ratio
        self.layout_output = LiltSelfOutput(config)
        config.hidden_size = ori_hidden_size

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads)
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(self, hidden_states: torch.Tensor, layout_inputs: torch.Tensor, attention_mask: Optional[torch.FloatTensor]=None, head_mask: Optional[torch.FloatTensor]=None, output_attentions: Optional[bool]=False) -> Tuple[torch.Tensor]:
        self_outputs = self.self(hidden_states, layout_inputs, attention_mask, head_mask, output_attentions)
        attention_output = self.output(self_outputs[0][0], hidden_states)
        layout_attention_output = self.layout_output(self_outputs[0][1], layout_inputs)
        outputs = ((attention_output, layout_attention_output),) + self_outputs[1:]
        return outputs