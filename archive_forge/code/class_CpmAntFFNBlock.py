import math
from typing import List, Optional, Tuple, Union
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss
from ...activations import ACT2FN
from ...modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from ...modeling_utils import PreTrainedModel
from ...utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward, logging
from .configuration_cpmant import CpmAntConfig
class CpmAntFFNBlock(nn.Module):

    def __init__(self, config: CpmAntConfig):
        super().__init__()
        self.layernorm_before_ffn = CpmAntLayerNorm(config)
        self.ffn = CpmAntFeedForward(config)
        if config.dropout_p:
            self.dropout = torch.nn.Dropout(config.dropout_p)
        else:
            self.dropout = None

    def forward(self, hidden_states: torch.Tensor):
        """
        Args:
            hidden_states (`torch.Tensor` of shape `(batch, len_seq, dim_model)`):
                Hidden states before feed forward layer.
        """
        ln_outputs = self.layernorm_before_ffn(hidden_states)
        outputs = self.ffn(ln_outputs)
        if self.dropout is not None:
            outputs = self.dropout(outputs)
        hidden_states = hidden_states + outputs
        return hidden_states