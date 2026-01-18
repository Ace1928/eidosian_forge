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
class PerceiverProjectionDecoder(PerceiverAbstractDecoder):
    """
    Baseline projection decoder (no cross-attention).

    Args:
        config ([`PerceiverConfig`]):
            Model configuration.
    """

    def __init__(self, config):
        super().__init__()
        self.classifier = nn.Linear(config.d_latents, config.num_labels)

    def decoder_query(self, inputs, modality_sizes=None, inputs_without_pos=None, subsampled_points=None):
        return None

    def forward(self, query: torch.Tensor, z: torch.FloatTensor, query_mask: Optional[torch.FloatTensor]=None) -> torch.FloatTensor:
        z = torch.mean(z, dim=1)
        logits = self.classifier(z)
        return logits