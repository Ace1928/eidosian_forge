import math
from typing import List, Optional, Tuple, Union
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss
from ...activations import ACT2FN
from ...modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from ...modeling_utils import PreTrainedModel
from ...utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward, logging
from .configuration_cpmant import CpmAntConfig
def _position_bucket(self, relative_position, num_buckets=32, max_distance=128):
    relative_buckets = 0
    num_buckets //= 2
    relative_buckets = (relative_position > 0).to(torch.int32) * num_buckets
    relative_position = torch.abs(relative_position)
    max_exact = num_buckets // 2
    is_small = relative_position < max_exact
    relative_postion_if_large = max_exact + (torch.log(relative_position.float() / max_exact) / math.log(max_distance / max_exact) * (num_buckets - max_exact)).to(torch.int32)
    relative_postion_if_large = torch.min(relative_postion_if_large, torch.full_like(relative_postion_if_large, num_buckets - 1))
    relative_buckets += torch.where(is_small, relative_position.to(torch.int32), relative_postion_if_large)
    return relative_buckets