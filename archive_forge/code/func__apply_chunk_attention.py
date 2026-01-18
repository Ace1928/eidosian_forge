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
from .configuration_seamless_m4t_v2 import SeamlessM4Tv2Config
def _apply_chunk_attention(self, attention_mask, hidden_states):
    """
        Creates a chunk attention mask. It creates a mask to prevent attention across chunks, ensuring that each
        position attends only to positions within its own chunk. If a left chunk overlap is specified
        (`speech_encoder_chunk_size` in the configuration), the attention mask is adjusted accordingly to allow each
        position to also attends the `speech_encoder_chunk_size - 1` previous chunks.
        """
    sequence_len = hidden_states.shape[1]
    chunk_indices = torch.arange(sequence_len, device=hidden_states.device)
    chunk_indices = torch.div(chunk_indices, self.config.speech_encoder_chunk_size).long()
    start_indices = torch.full_like(chunk_indices, 0)
    if self.config.speech_encoder_left_chunk_num >= 0:
        start_indices = (chunk_indices - self.config.speech_encoder_left_chunk_num).clamp_(min=0)
        start_indices = start_indices * self.config.speech_encoder_chunk_size
        start_indices = start_indices
    start_indices = start_indices.unsqueeze(1).expand(-1, sequence_len)
    end_indices = ((chunk_indices + 1) * self.config.speech_encoder_chunk_size).clamp_(max=sequence_len)
    end_indices = end_indices.unsqueeze(1).expand(-1, sequence_len)
    indices = torch.arange(sequence_len, device=hidden_states.device).unsqueeze(0).expand(sequence_len, -1)
    chunk_mask = (indices < start_indices) | (indices >= end_indices)
    chunk_mask = chunk_mask.unsqueeze(0).unsqueeze(0)
    attention_mask = chunk_mask if attention_mask is None else attention_mask.bool() | chunk_mask
    attention_mask = attention_mask.to(dtype=hidden_states.dtype)
    return attention_mask