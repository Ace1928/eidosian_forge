import math
from typing import List, Optional, Tuple, Union
import numpy as np
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, L1Loss
from ...activations import ACT2FN
from ...integrations.deepspeed import is_deepspeed_zero3_enabled
from ...modeling_attn_mask_utils import _prepare_4d_attention_mask, _prepare_4d_causal_attention_mask
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from .configuration_speecht5 import SpeechT5Config, SpeechT5HifiGanConfig
def _mask_hidden_states(self, hidden_states: torch.FloatTensor, mask_time_indices: Optional[torch.FloatTensor]=None, attention_mask: Optional[torch.LongTensor]=None):
    """
        Masks extracted features along time axis and/or along feature axis according to
        [SpecAugment](https://arxiv.org/abs/1904.08779).
        """
    if not getattr(self.config, 'apply_spec_augment', True):
        return hidden_states
    batch_size, sequence_length, hidden_size = hidden_states.size()
    if mask_time_indices is not None:
        hidden_states[mask_time_indices] = self.masked_spec_embed.to(hidden_states.dtype)
    elif self.config.mask_time_prob > 0 and self.training:
        mask_time_indices = _compute_mask_indices((batch_size, sequence_length), mask_prob=self.config.mask_time_prob, mask_length=self.config.mask_time_length, attention_mask=attention_mask, min_masks=self.config.mask_time_min_masks)
        mask_time_indices = torch.tensor(mask_time_indices, device=hidden_states.device, dtype=torch.bool)
        hidden_states[mask_time_indices] = self.masked_spec_embed.to(hidden_states.dtype)
    if self.config.mask_feature_prob > 0 and self.training:
        mask_feature_indices = _compute_mask_indices((batch_size, hidden_size), mask_prob=self.config.mask_feature_prob, mask_length=self.config.mask_feature_length, min_masks=self.config.mask_feature_min_masks)
        mask_feature_indices = torch.tensor(mask_feature_indices, device=hidden_states.device, dtype=torch.bool)
        mask_feature_indices = mask_feature_indices[:, None].expand(-1, sequence_length, -1)
        hidden_states[mask_feature_indices] = 0
    return hidden_states