import math
from typing import Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import gelu
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward, logging
from .configuration_ibert import IBertConfig
from .quant_modules import IntGELU, IntLayerNorm, IntSoftmax, QuantAct, QuantEmbedding, QuantLinear
class IBertLayer(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.quant_mode = config.quant_mode
        self.act_bit = 8
        self.seq_len_dim = 1
        self.attention = IBertAttention(config)
        self.intermediate = IBertIntermediate(config)
        self.output = IBertOutput(config)
        self.pre_intermediate_act = QuantAct(self.act_bit, quant_mode=self.quant_mode)
        self.pre_output_act = QuantAct(self.act_bit, quant_mode=self.quant_mode)

    def forward(self, hidden_states, hidden_states_scaling_factor, attention_mask=None, head_mask=None, output_attentions=False):
        self_attention_outputs, self_attention_outputs_scaling_factor = self.attention(hidden_states, hidden_states_scaling_factor, attention_mask, head_mask, output_attentions=output_attentions)
        attention_output = self_attention_outputs[0]
        attention_output_scaling_factor = self_attention_outputs_scaling_factor[0]
        outputs = self_attention_outputs[1:]
        layer_output, layer_output_scaling_factor = self.feed_forward_chunk(attention_output, attention_output_scaling_factor)
        outputs = (layer_output,) + outputs
        return outputs

    def feed_forward_chunk(self, attention_output, attention_output_scaling_factor):
        attention_output, attention_output_scaling_factor = self.pre_intermediate_act(attention_output, attention_output_scaling_factor)
        intermediate_output, intermediate_output_scaling_factor = self.intermediate(attention_output, attention_output_scaling_factor)
        intermediate_output, intermediate_output_scaling_factor = self.pre_output_act(intermediate_output, intermediate_output_scaling_factor)
        layer_output, layer_output_scaling_factor = self.output(intermediate_output, intermediate_output_scaling_factor, attention_output, attention_output_scaling_factor)
        return (layer_output, layer_output_scaling_factor)