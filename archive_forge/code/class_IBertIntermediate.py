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
class IBertIntermediate(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.quant_mode = config.quant_mode
        self.act_bit = 8
        self.weight_bit = 8
        self.bias_bit = 32
        self.dense = QuantLinear(config.hidden_size, config.intermediate_size, bias=True, weight_bit=self.weight_bit, bias_bit=self.bias_bit, quant_mode=self.quant_mode, per_channel=True)
        if config.hidden_act != 'gelu':
            raise ValueError("I-BERT only supports 'gelu' for `config.hidden_act`")
        self.intermediate_act_fn = IntGELU(quant_mode=self.quant_mode, force_dequant=config.force_dequant)
        self.output_activation = QuantAct(self.act_bit, quant_mode=self.quant_mode)

    def forward(self, hidden_states, hidden_states_scaling_factor):
        hidden_states, hidden_states_scaling_factor = self.dense(hidden_states, hidden_states_scaling_factor)
        hidden_states, hidden_states_scaling_factor = self.intermediate_act_fn(hidden_states, hidden_states_scaling_factor)
        hidden_states, hidden_states_scaling_factor = self.output_activation(hidden_states, hidden_states_scaling_factor)
        return (hidden_states, hidden_states_scaling_factor)