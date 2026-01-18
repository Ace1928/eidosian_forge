import copy
import math
import os
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import apply_chunking_to_forward, find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import (
from .configuration_canine import CanineConfig
def _create_3d_attention_mask_from_input_mask(self, from_tensor, to_mask):
    """
        Create 3D attention mask from a 2D tensor mask.

        Args:
            from_tensor: 2D or 3D Tensor of shape [batch_size, from_seq_length, ...].
            to_mask: int32 Tensor of shape [batch_size, to_seq_length].

        Returns:
            float Tensor of shape [batch_size, from_seq_length, to_seq_length].
        """
    batch_size, from_seq_length = (from_tensor.shape[0], from_tensor.shape[1])
    to_seq_length = to_mask.shape[1]
    to_mask = torch.reshape(to_mask, (batch_size, 1, to_seq_length)).float()
    broadcast_ones = torch.ones(size=(batch_size, from_seq_length, 1), dtype=torch.float32, device=to_mask.device)
    mask = broadcast_ones * to_mask
    return mask