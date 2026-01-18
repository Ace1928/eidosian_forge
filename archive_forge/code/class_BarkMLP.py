import math
from typing import Dict, Optional, Tuple, Union
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from ...generation.logits_process import (
from ...modeling_attn_mask_utils import _prepare_4d_attention_mask
from ...modeling_outputs import CausalLMOutputWithPast, MaskedLMOutput
from ...modeling_utils import PreTrainedModel, get_parameter_device
from ...utils import (
from ..auto import AutoModel
from .configuration_bark import (
from .generation_configuration_bark import (
class BarkMLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.in_proj = nn.Linear(config.hidden_size, 4 * config.hidden_size, bias=config.bias)
        self.out_proj = nn.Linear(4 * config.hidden_size, config.hidden_size, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)
        self.gelu = nn.GELU()

    def forward(self, hidden_states):
        hidden_states = self.in_proj(hidden_states)
        hidden_states = self.gelu(hidden_states)
        hidden_states = self.out_proj(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states