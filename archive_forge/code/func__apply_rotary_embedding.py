import copy
import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import Tensor, nn
from torch.nn import CrossEntropyLoss
from ...activations import ACT2FN
from ...deepspeed import is_deepspeed_zero3_enabled
from ...modeling_attn_mask_utils import _prepare_4d_attention_mask, _prepare_4d_causal_attention_mask
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...utils import (
from .configuration_seamless_m4t import SeamlessM4TConfig
def _apply_rotary_embedding(self, hidden_states, relative_position_embeddings):
    batch_size, sequence_length, hidden_size = hidden_states.size()
    hidden_states = hidden_states.view(batch_size, sequence_length, self.num_heads, self.head_size)
    cos = relative_position_embeddings[0, :sequence_length, ...]
    sin = relative_position_embeddings[1, :sequence_length, ...]
    hidden_states = hidden_states.transpose(0, 1)
    rotated_states_begin = hidden_states[..., :self.head_size // 2]
    rotated_states_end = hidden_states[..., self.head_size // 2:]
    rotated_states = torch.cat((-rotated_states_end, rotated_states_begin), dim=rotated_states_begin.ndim - 1)
    hidden_states = hidden_states * cos + rotated_states * sin
    hidden_states = hidden_states.transpose(0, 1)
    hidden_states = hidden_states.view(batch_size, sequence_length, self.num_heads * self.head_size)
    return hidden_states