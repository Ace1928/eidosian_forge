import math
from typing import Optional, Tuple, Union
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss
from ...activations import ACT2FN
from ...modeling_attn_mask_utils import _prepare_4d_causal_attention_mask, _prepare_4d_causal_attention_mask_for_sdpa
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...utils import (
from .configuration_whisper import WhisperConfig
from .generation_whisper import WhisperGenerationMixin
class WhisperPositionalEmbedding(nn.Embedding):

    def __init__(self, num_positions: int, embedding_dim: int, padding_idx: Optional[int]=None):
        super().__init__(num_positions, embedding_dim)

    def forward(self, input_ids, past_key_values_length=0, position_ids=None):
        if position_ids is None:
            return self.weight[past_key_values_length:past_key_values_length + input_ids.shape[1]]
        else:
            return self.weight[position_ids]