import math
from typing import Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from ....activations import ACT2FN
from ....file_utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward
from ....integrations.deepspeed import is_deepspeed_zero3_enabled
from ....modeling_attn_mask_utils import _prepare_4d_attention_mask
from ....modeling_outputs import BaseModelOutput, CausalLMOutput
from ....modeling_utils import (
from ....utils import logging
from .configuration_mctct import MCTCTConfig
class MCTCTLayer(nn.Module):

    def __init__(self, config: MCTCTConfig):
        super().__init__()
        self.seq_len_dim = 1
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.intermediate = MCTCTIntermediate(config)
        self.attention = MCTCTAttention(config)
        self.is_decoder = config.is_decoder
        self.output = MCTCTOutput(config)

    def forward(self, hidden_states, attention_mask=None, head_mask=None, output_attentions=False):
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