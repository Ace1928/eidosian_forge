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
from .configuration_wav2vec2_conformer import Wav2Vec2ConformerConfig
class Wav2Vec2ConformerGumbelVectorQuantizer(nn.Module):
    """
    Vector quantization using gumbel softmax. See `[CATEGORICAL REPARAMETERIZATION WITH
    GUMBEL-SOFTMAX](https://arxiv.org/pdf/1611.01144.pdf) for more information.
    """

    def __init__(self, config):
        super().__init__()
        self.num_groups = config.num_codevector_groups
        self.num_vars = config.num_codevectors_per_group
        if config.codevector_dim % self.num_groups != 0:
            raise ValueError(f'`config.codevector_dim {config.codevector_dim} must be divisible by `config.num_codevector_groups` {self.num_groups} for concatenation')
        self.codevectors = nn.Parameter(torch.FloatTensor(1, self.num_groups * self.num_vars, config.codevector_dim // self.num_groups))
        self.weight_proj = nn.Linear(config.conv_dim[-1], self.num_groups * self.num_vars)
        self.temperature = 2

    @staticmethod
    def _compute_perplexity(probs, mask=None):
        if mask is not None:
            mask_extended = mask.flatten()[:, None, None].expand(probs.shape)
            probs = torch.where(mask_extended, probs, torch.zeros_like(probs))
            marginal_probs = probs.sum(dim=0) / mask.sum()
        else:
            marginal_probs = probs.mean(dim=0)
        perplexity = torch.exp(-torch.sum(marginal_probs * torch.log(marginal_probs + 1e-07), dim=-1)).sum()
        return perplexity

    def forward(self, hidden_states, mask_time_indices=None):
        batch_size, sequence_length, hidden_size = hidden_states.shape
        hidden_states = self.weight_proj(hidden_states)
        hidden_states = hidden_states.view(batch_size * sequence_length * self.num_groups, -1)
        if self.training:
            codevector_probs = nn.functional.gumbel_softmax(hidden_states.float(), tau=self.temperature, hard=True).type_as(hidden_states)
            codevector_soft_dist = torch.softmax(hidden_states.view(batch_size * sequence_length, self.num_groups, -1).float(), dim=-1)
            perplexity = self._compute_perplexity(codevector_soft_dist, mask_time_indices)
        else:
            codevector_idx = hidden_states.argmax(dim=-1)
            codevector_probs = hidden_states.new_zeros(hidden_states.shape).scatter_(-1, codevector_idx.view(-1, 1), 1.0)
            codevector_probs = codevector_probs.view(batch_size * sequence_length, self.num_groups, -1)
            perplexity = self._compute_perplexity(codevector_probs, mask_time_indices)
        codevector_probs = codevector_probs.view(batch_size * sequence_length, -1)
        codevectors_per_group = codevector_probs.unsqueeze(-1) * self.codevectors
        codevectors = codevectors_per_group.view(batch_size * sequence_length, self.num_groups, self.num_vars, -1)
        codevectors = codevectors.sum(-2).view(batch_size, sequence_length, -1)
        return (codevectors, perplexity)