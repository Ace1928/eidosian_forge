import math
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import numpy as np
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss
from ...activations import ACT2FN
from ...integrations.deepspeed import is_deepspeed_zero3_enabled
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...utils import (
from .configuration_unispeech_sat import UniSpeechSatConfig
class UniSpeechSatGumbelVectorQuantizer(nn.Module):
    """
    Vector quantization using gumbel softmax. See [CATEGORICAL REPARAMETERIZATION WITH
    GUMBEL-SOFTMAX](https://arxiv.org/pdf/1611.01144.pdf) for more information.
    """

    def __init__(self, config):
        super().__init__()
        self.num_groups = config.num_codevector_groups
        self.num_vars = config.num_codevectors_per_group
        if config.codevector_dim % self.num_groups != 0:
            raise ValueError(f'`config.codevector_dim {config.codevector_dim} must be divisible by `config.num_codevector_groups` {self.num_groups} for concatenation')
        self.codevectors = nn.Parameter(torch.FloatTensor(1, self.num_groups * self.num_vars, config.codevector_dim // self.num_groups))
        self.weight_proj = nn.Linear(config.hidden_size, self.num_groups * self.num_vars)
        self.temperature = 2

    @staticmethod
    def _compute_perplexity(probs, mask=None):
        marginal_probs = probs.mean(dim=0)
        perplexity = torch.exp(-torch.sum(marginal_probs * torch.log(marginal_probs + 1e-07), dim=-1)).sum()
        return perplexity

    def forward(self, hidden_states):
        batch_size, sequence_length, hidden_size = hidden_states.shape
        hidden_states = self.weight_proj(hidden_states)
        hidden_states = hidden_states.view(batch_size * sequence_length * self.num_groups, -1)
        if self.training:
            codevector_probs = nn.functional.gumbel_softmax(hidden_states.float(), tau=self.temperature, hard=True).type_as(hidden_states)
            codevector_soft_dist = torch.softmax(hidden_states.view(batch_size * sequence_length, self.num_groups, -1).float(), dim=-1)
            perplexity = self._compute_perplexity(codevector_soft_dist)
        else:
            codevector_idx = hidden_states.argmax(dim=-1)
            codevector_probs = hidden_states.new_zeros(*hidden_states.shape).scatter_(-1, codevector_idx.view(-1, 1), 1.0)
            codevector_probs = codevector_probs.view(batch_size * sequence_length, self.num_groups, -1)
            perplexity = self._compute_perplexity(codevector_probs)
        codevector_probs = codevector_probs.view(batch_size * sequence_length, -1)
        codevectors_per_group = codevector_probs.unsqueeze(-1) * self.codevectors
        codevectors = codevectors_per_group.view(batch_size * sequence_length, self.num_groups, self.num_vars, -1)
        codevectors = codevectors.sum(-2).view(batch_size, sequence_length, -1)
        return (codevectors, perplexity)