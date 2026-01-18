import collections
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import Tensor, nn
from ...activations import ACT2FN
from ...modeling_outputs import BaseModelOutput
from ...modeling_utils import PreTrainedModel
from ...utils import ModelOutput, add_start_docstrings, add_start_docstrings_to_model_forward, logging
from .configuration_sam import SamConfig, SamMaskDecoderConfig, SamPromptEncoderConfig, SamVisionConfig
def add_decomposed_rel_pos(self, attn: torch.Tensor, query: torch.Tensor, rel_pos_h: torch.Tensor, rel_pos_w: torch.Tensor, q_size: Tuple[int, int], k_size: Tuple[int, int]) -> torch.Tensor:
    """
        Calculate decomposed Relative Positional Embeddings from :paper:`mvitv2`.
        https://github.com/facebookresearch/mvit/blob/19786631e330df9f3622e5402b4a419a263a2c80/mvit/models/attention.py

        Args:
            attn (`torch.Tensor`):
                attention map.
            query (`torch.Tensor`):
                query q in the attention layer with shape (batch_size, query_height * query_width, channel).
            rel_pos_h (`torch.Tensor`):
                relative position embeddings (Lh, channel) for height axis.
            rel_pos_w (`torch.Tensor`):
                relative position embeddings (Lw, channel) for width axis.
            q_size (tuple):
                spatial sequence size of query q with (query_height, query_width).
            k_size (tuple):
                spatial sequence size of key k with (key_height, key_width).

        Returns:
            attn (`torch.Tensor`):
                attention map with added relative positional embeddings.
        """
    query_height, query_width = q_size
    key_height, key_width = k_size
    relative_position_height = self.get_rel_pos(query_height, key_height, rel_pos_h)
    relative_position_width = self.get_rel_pos(query_width, key_width, rel_pos_w)
    batch_size, _, dim = query.shape
    reshaped_query = query.reshape(batch_size, query_height, query_width, dim)
    rel_h = torch.einsum('bhwc,hkc->bhwk', reshaped_query, relative_position_height)
    rel_w = torch.einsum('bhwc,wkc->bhwk', reshaped_query, relative_position_width)
    attn = attn.reshape(batch_size, query_height, query_width, key_height, key_width)
    attn = attn + rel_h[:, :, :, :, None] + rel_w[:, :, :, None, :]
    attn = attn.reshape(batch_size, query_height * query_width, key_height * key_width)
    return attn