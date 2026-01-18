import copy
import math
import os
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import apply_chunking_to_forward, find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import (
from .configuration_canine import CanineConfig
class CanineLayer(nn.Module):

    def __init__(self, config, local, always_attend_to_first_position, first_position_attends_to_all, attend_from_chunk_width, attend_from_chunk_stride, attend_to_chunk_width, attend_to_chunk_stride):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = CanineAttention(config, local, always_attend_to_first_position, first_position_attends_to_all, attend_from_chunk_width, attend_from_chunk_stride, attend_to_chunk_width, attend_to_chunk_stride)
        self.intermediate = CanineIntermediate(config)
        self.output = CanineOutput(config)

    def forward(self, hidden_states: Tuple[torch.FloatTensor], attention_mask: Optional[torch.FloatTensor]=None, head_mask: Optional[torch.FloatTensor]=None, output_attentions: Optional[bool]=False) -> Tuple[torch.FloatTensor, Optional[torch.FloatTensor]]:
        self_attention_outputs = self.attention(hidden_states, attention_mask, head_mask, output_attentions=output_attentions)
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]
        layer_output = apply_chunking_to_forward(self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output)
        outputs = (layer_output,) + outputs
        return outputs

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output