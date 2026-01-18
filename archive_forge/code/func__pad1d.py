import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from ...modeling_utils import PreTrainedModel
from ...utils import (
from .configuration_encodec import EncodecConfig
@staticmethod
def _pad1d(hidden_states: torch.Tensor, paddings: Tuple[int, int], mode: str='zero', value: float=0.0):
    """Tiny wrapper around torch.nn.functional.pad, just to allow for reflect padding on small input.
        If this is the case, we insert extra 0 padding to the right before the reflection happens.
        """
    length = hidden_states.shape[-1]
    padding_left, padding_right = paddings
    if not mode == 'reflect':
        return nn.functional.pad(hidden_states, paddings, mode, value)
    max_pad = max(padding_left, padding_right)
    extra_pad = 0
    if length <= max_pad:
        extra_pad = max_pad - length + 1
        hidden_states = nn.functional.pad(hidden_states, (0, extra_pad))
    padded = nn.functional.pad(hidden_states, paddings, mode, value)
    end = padded.shape[-1] - extra_pad
    return padded[..., :end]