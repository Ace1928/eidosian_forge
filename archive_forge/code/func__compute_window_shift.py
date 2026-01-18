import collections.abc
import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from ...activations import ACT2FN
from ...modeling_outputs import BaseModelOutput, ImageSuperResolutionOutput
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import find_pruneable_heads_and_indices, meshgrid, prune_linear_layer
from ...utils import (
from .configuration_swin2sr import Swin2SRConfig
def _compute_window_shift(self, target_window_size, target_shift_size) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    window_size = [r if r <= w else w for r, w in zip(self.input_resolution, target_window_size)]
    shift_size = [0 if r <= w else s for r, w, s in zip(self.input_resolution, window_size, target_shift_size)]
    return (window_size, shift_size)