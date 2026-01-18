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
class PerceiverTextPreprocessor(AbstractPreprocessor):
    """
    Text preprocessing for Perceiver Encoder. Can be used to embed `inputs` and add positional encodings.

    The dimensionality of the embeddings is determined by the `d_model` attribute of the configuration.

    Args:
        config ([`PerceiverConfig`]):
            Model configuration.
    """

    def __init__(self, config: PerceiverConfig) -> None:
        super().__init__()
        self.config = config
        self.embeddings = nn.Embedding(num_embeddings=config.vocab_size, embedding_dim=config.d_model)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.d_model)

    @property
    def num_channels(self) -> int:
        return self.config.d_model

    def forward(self, inputs: torch.LongTensor, pos: Optional[torch.Tensor]=None, network_input_is_1d: bool=True):
        embeddings_without_pos = self.embeddings(inputs)
        seq_length = inputs.shape[1]
        position_ids = torch.arange(0, seq_length, device=inputs.device)
        embeddings = embeddings_without_pos + self.position_embeddings(position_ids)
        return (embeddings, None, embeddings_without_pos)