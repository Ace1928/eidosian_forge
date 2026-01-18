import math
from collections import OrderedDict
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss
from ...activations import ACT2FN, QuickGELUActivation
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel, apply_chunking_to_forward
from ...pytorch_utils import find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from .configuration_bridgetower import BridgeTowerConfig, BridgeTowerTextConfig, BridgeTowerVisionConfig
class BridgeTowerBertCrossLayer(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = BridgeTowerAttention(config)
        self.is_decoder = config.is_decoder
        self.add_cross_attention = config.add_cross_attention
        self.crossattention = BridgeTowerAttention(config)
        self.intermediate = BridgeTowerIntermediate(config)
        self.output = BridgeTowerOutput(config)

    def forward(self, hidden_states, encoder_hidden_states, attention_mask=None, head_mask=None, encoder_attention_mask=None, past_key_value=None, output_attentions=False):
        self_attention_outputs = self.attention(hidden_states, attention_mask=attention_mask, head_mask=None, output_attentions=output_attentions, past_key_value=None)
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]
        cross_attention_outputs = self.crossattention(attention_output, attention_mask=attention_mask, head_mask=head_mask, encoder_hidden_states=encoder_hidden_states, encoder_attention_mask=encoder_attention_mask, past_key_value=past_key_value, output_attentions=output_attentions)
        attention_output = cross_attention_outputs[0]
        outputs = outputs + cross_attention_outputs[1:-1]
        layer_output = apply_chunking_to_forward(self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output)
        outputs = (layer_output,) + outputs
        return outputs

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output