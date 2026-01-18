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
class PerceiverFourierPositionEncoding(PerceiverAbstractPositionEncoding):
    """Fourier (Sinusoidal) position encoding."""

    def __init__(self, num_bands, max_resolution, concat_pos=True, sine_only=False):
        super().__init__()
        self.num_bands = num_bands
        self.max_resolution = max_resolution
        self.concat_pos = concat_pos
        self.sine_only = sine_only

    @property
    def num_dimensions(self) -> int:
        return len(self.max_resolution)

    def output_size(self):
        """Returns size of positional encodings last dimension."""
        num_dims = len(self.max_resolution)
        encoding_size = self.num_bands * num_dims
        if not self.sine_only:
            encoding_size *= 2
        if self.concat_pos:
            encoding_size += self.num_dimensions
        return encoding_size

    def forward(self, index_dims: List[int], batch_size: int, device: torch.device, dtype: torch.dtype, pos: torch.FloatTensor=None) -> torch.FloatTensor:
        pos = _check_or_build_spatial_positions(pos, index_dims, batch_size)
        fourier_pos_enc = generate_fourier_features(pos, num_bands=self.num_bands, max_resolution=self.max_resolution, concat_pos=self.concat_pos, sine_only=self.sine_only).to(device=device, dtype=dtype)
        return fourier_pos_enc