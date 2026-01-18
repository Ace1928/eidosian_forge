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
class LiltLayer(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = LiltAttention(config)
        self.intermediate = LiltIntermediate(config)
        self.output = LiltOutput(config)
        ori_hidden_size = config.hidden_size
        ori_intermediate_size = config.intermediate_size
        config.hidden_size = config.hidden_size // config.channel_shrink_ratio
        config.intermediate_size = config.intermediate_size // config.channel_shrink_ratio
        self.layout_intermediate = LiltIntermediate(config)
        self.layout_output = LiltOutput(config)
        config.hidden_size = ori_hidden_size
        config.intermediate_size = ori_intermediate_size

    def forward(self, hidden_states: torch.Tensor, layout_inputs: torch.Tensor, attention_mask: Optional[torch.FloatTensor]=None, head_mask: Optional[torch.FloatTensor]=None, output_attentions: Optional[bool]=False) -> Tuple[torch.Tensor]:
        self_attention_outputs = self.attention(hidden_states, layout_inputs, attention_mask, head_mask, output_attentions=output_attentions)
        attention_output = self_attention_outputs[0][0]
        layout_attention_output = self_attention_outputs[0][1]
        outputs = self_attention_outputs[1:]
        layer_output = apply_chunking_to_forward(self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output)
        layout_layer_output = apply_chunking_to_forward(self.layout_feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, layout_attention_output)
        outputs = ((layer_output, layout_layer_output),) + outputs
        return outputs

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output

    def layout_feed_forward_chunk(self, attention_output):
        intermediate_output = self.layout_intermediate(attention_output)
        layer_output = self.layout_output(intermediate_output, attention_output)
        return layer_output