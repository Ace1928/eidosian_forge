import math
import warnings
from typing import Optional, Tuple, Union
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss
from ...activations import ACT2FN
from ...integrations.deepspeed import is_deepspeed_zero3_enabled
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...utils import (
from .configuration_wavlm import WavLMConfig
def _relative_positions_bucket(self, relative_positions: torch.FloatTensor) -> torch.FloatTensor:
    num_buckets = self.num_buckets // 2
    relative_buckets = (relative_positions > 0).to(torch.long) * num_buckets
    relative_positions = torch.abs(relative_positions)
    max_exact = num_buckets // 2
    is_small = relative_positions < max_exact
    relative_positions_if_large = torch.log(relative_positions.float() / max_exact)
    relative_positions_if_large = relative_positions_if_large / math.log(self.max_distance / max_exact)
    relative_positions_if_large = relative_positions_if_large * (num_buckets - max_exact)
    relative_position_if_large = (max_exact + relative_positions_if_large).to(torch.long)
    relative_position_if_large = torch.min(relative_position_if_large, torch.full_like(relative_position_if_large, num_buckets - 1))
    relative_buckets += torch.where(is_small, relative_positions, relative_position_if_large)
    return relative_buckets