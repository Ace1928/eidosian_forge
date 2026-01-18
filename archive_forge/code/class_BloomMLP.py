import math
import warnings
from typing import Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, LayerNorm, MSELoss
from torch.nn import functional as F
from ...file_utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward
from ...modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...utils import logging
from .configuration_bloom import BloomConfig
class BloomMLP(nn.Module):

    def __init__(self, config: BloomConfig):
        super().__init__()
        hidden_size = config.hidden_size
        self.pretraining_tp = config.pretraining_tp
        self.slow_but_exact = config.slow_but_exact
        self.dense_h_to_4h = nn.Linear(hidden_size, 4 * hidden_size)
        self.gelu_impl = BloomGelu()
        self.dense_4h_to_h = nn.Linear(4 * hidden_size, hidden_size)
        self.hidden_dropout = config.hidden_dropout

    def forward(self, hidden_states: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        hidden_states = self.gelu_impl(self.dense_h_to_4h(hidden_states))
        if self.pretraining_tp > 1 and self.slow_but_exact:
            intermediate_output = torch.zeros_like(residual)
            slices = self.dense_4h_to_h.weight.shape[-1] / self.pretraining_tp
            for i in range(self.pretraining_tp):
                intermediate_output = intermediate_output + F.linear(hidden_states[:, :, int(i * slices):int((i + 1) * slices)], self.dense_4h_to_h.weight[:, int(i * slices):int((i + 1) * slices)])
        else:
            intermediate_output = self.dense_4h_to_h(hidden_states)
        output = dropout_add(intermediate_output, residual, self.hidden_dropout, self.training)
        return output