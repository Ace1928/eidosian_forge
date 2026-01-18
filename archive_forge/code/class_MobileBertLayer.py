import math
import os
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import (
from .configuration_mobilebert import MobileBertConfig
class MobileBertLayer(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.use_bottleneck = config.use_bottleneck
        self.num_feedforward_networks = config.num_feedforward_networks
        self.attention = MobileBertAttention(config)
        self.intermediate = MobileBertIntermediate(config)
        self.output = MobileBertOutput(config)
        if self.use_bottleneck:
            self.bottleneck = Bottleneck(config)
        if config.num_feedforward_networks > 1:
            self.ffn = nn.ModuleList([FFNLayer(config) for _ in range(config.num_feedforward_networks - 1)])

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.FloatTensor]=None, head_mask: Optional[torch.FloatTensor]=None, output_attentions: Optional[bool]=None) -> Tuple[torch.Tensor]:
        if self.use_bottleneck:
            query_tensor, key_tensor, value_tensor, layer_input = self.bottleneck(hidden_states)
        else:
            query_tensor, key_tensor, value_tensor, layer_input = [hidden_states] * 4
        self_attention_outputs = self.attention(query_tensor, key_tensor, value_tensor, layer_input, attention_mask, head_mask, output_attentions=output_attentions)
        attention_output = self_attention_outputs[0]
        s = (attention_output,)
        outputs = self_attention_outputs[1:]
        if self.num_feedforward_networks != 1:
            for i, ffn_module in enumerate(self.ffn):
                attention_output = ffn_module(attention_output)
                s += (attention_output,)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output, hidden_states)
        outputs = (layer_output,) + outputs + (torch.tensor(1000), query_tensor, key_tensor, value_tensor, layer_input, attention_output, intermediate_output) + s
        return outputs