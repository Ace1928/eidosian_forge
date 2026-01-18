import abc
import math
from dataclasses import dataclass
from functools import reduce
from operator import __add__
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple, Union
import numpy as np
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_outputs import BaseModelOutputWithCrossAttentions
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import apply_chunking_to_forward, find_pruneable_heads_and_indices, meshgrid, prune_linear_layer
from ...utils import (
from .configuration_perceiver import PerceiverConfig
class PerceiverEmbeddingDecoder(nn.Module):
    """
    Module to decode embeddings (for masked language modeling).

    Args:
        config ([`PerceiverConfig`]):
            Model configuration.
    """

    def __init__(self, config: PerceiverConfig) -> None:
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.bias = nn.Parameter(torch.zeros(self.vocab_size))

    def forward(self, hidden_states: torch.Tensor, embedding_layer: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, d_model = hidden_states.shape
        output = torch.matmul(hidden_states.reshape([-1, d_model]), embedding_layer.weight.transpose(0, 1))
        output = output + self.bias
        return output.reshape([batch_size, seq_len, self.vocab_size])