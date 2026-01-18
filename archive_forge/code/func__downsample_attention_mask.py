import copy
import math
import os
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import apply_chunking_to_forward, find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import (
from .configuration_canine import CanineConfig
def _downsample_attention_mask(self, char_attention_mask: torch.Tensor, downsampling_rate: int):
    """Downsample 2D character attention mask to 2D molecule attention mask using MaxPool1d layer."""
    batch_size, char_seq_len = char_attention_mask.shape
    poolable_char_mask = torch.reshape(char_attention_mask, (batch_size, 1, char_seq_len))
    pooled_molecule_mask = torch.nn.MaxPool1d(kernel_size=downsampling_rate, stride=downsampling_rate)(poolable_char_mask.float())
    molecule_attention_mask = torch.squeeze(pooled_molecule_mask, dim=-1)
    return molecule_attention_mask