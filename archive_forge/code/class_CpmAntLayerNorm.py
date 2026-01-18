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
class CpmAntLayerNorm(nn.Module):
    """
    We use Root Mean Square (RMS) Layer Normalization, please see https://arxiv.org/abs/1910.07467 for details."
    """

    def __init__(self, config: CpmAntConfig):
        super().__init__()
        self.eps = config.eps
        self.dim_norm = config.hidden_size
        self.weight = nn.Parameter(torch.empty(config.hidden_size))

    def forward(self, hidden_states: torch.Tensor):
        """
        Args:
            hidden_states (`torch.Tensor` of shape `(batch, seq_len, dim_in)`)
        """
        if hidden_states.size(-1) != self.dim_norm:
            raise AssertionError('hidden_states.size(-1) != self.dim_norm')
        old_dtype = hidden_states.dtype
        variance = hidden_states.to(torch.float32).pow(2).mean(dim=-1, keepdim=True)
        hidden_states = (hidden_states * torch.rsqrt(variance + self.eps)).to(old_dtype) * self.weight
        return hidden_states