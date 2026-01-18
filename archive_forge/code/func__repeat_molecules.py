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
def _repeat_molecules(self, molecules: torch.Tensor, char_seq_length: torch.Tensor) -> torch.Tensor:
    """Repeats molecules to make them the same length as the char sequence."""
    rate = self.config.downsampling_rate
    molecules_without_extra_cls = molecules[:, 1:, :]
    repeated = torch.repeat_interleave(molecules_without_extra_cls, repeats=rate, dim=-2)
    last_molecule = molecules[:, -1:, :]
    remainder_length = torch.fmod(torch.tensor(char_seq_length), torch.tensor(rate)).item()
    remainder_repeated = torch.repeat_interleave(last_molecule, repeats=remainder_length + rate, dim=-2)
    return torch.cat([repeated, remainder_repeated], dim=-2)