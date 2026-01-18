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
def _get_extra_padding_for_conv1d(hidden_states: torch.Tensor, kernel_size: int, stride: int, padding_total: int=0) -> int:
    """See `pad_for_conv1d`."""
    length = hidden_states.shape[-1]
    n_frames = (length - kernel_size + padding_total) / stride + 1
    ideal_length = (math.ceil(n_frames) - 1) * stride + (kernel_size - padding_total)
    return ideal_length - length